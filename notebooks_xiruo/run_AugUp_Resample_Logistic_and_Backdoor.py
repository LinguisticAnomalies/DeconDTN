import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, choices=["SHAC", "HateSpeech", "CD"], help="dataset name"
)
parser.add_argument(
    "--transform",
    type=str,
    choices=["binaryUnigram", "Sentence-BERT"],
    help="choose what kind of text representation to use",
)
parser.add_argument(
    "--constraintCy",
    action="store_true",
    help="If use constraint on Cy to enforce Cy train == Cy test.",
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="On which GPU to run",
)
args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import sys

sys.path.append("../src")

from copy import deepcopy
import math
import pandas as pd
import numpy as np

import random
import itertools
from sklearn import metrics
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer

import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from utils import number_split, create_mix, appendMetrics
from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC
from custom_distance import KL
from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_CD import load_cd


def confusion_matrix_probs(y_true, y_pred):

    idx = y_true == 0
    n_t0_p0 = sum(1 - y_pred[:, 1][idx])
    n_t0_p1 = sum(y_pred[:, 1][idx])

    n_t1_p0 = sum(1 - y_pred[:, 1][~idx])
    n_t1_p1 = sum(y_pred[:, 1][~idx])

    return np.array([[n_t0_p0, n_t1_p0], [n_t0_p1, n_t1_p1]])


######## Load Data
if args.dataset == "SHAC":
    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x["Drug"] else 0, axis=1)
    df_shac["dfSource"] = df_shac["location"]

    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

    n_test = 200
elif args.dataset == "HateSpeech":
    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()

    # n_test = 1000
    n_test = 200
elif args.dataset == "CD":
    df_all = load_cd()
    df_avh = df_all["avh"]
    df_r56 = df_all["r56"]

    n_test = 200


else:
    sys.exit("no such dataset for processing")

######## Split
train_test_ratio = 4

if args.constraintCy:
    p_pos_train_z0_ls = np.arange(
        0, 1, 0.1
    )  # probability of training set examples drawn from site/domain z0 being positive
    p_pos_train_z1_ls = np.arange(
        0, 1, 0.1
    )  # probability of test set examples drawn from site/domain z1 being positive

    p_mix_z1_ls = np.arange(0, 1, 0.05)

    alpha_test_ls = np.concatenate(
        [np.float_power(10, np.linspace(start=-2, stop=2, num=50)), [0.2, 1, 5]]
    )

    valid_full_settings = []
    for combination in itertools.product(
        p_pos_train_z0_ls, p_pos_train_z1_ls, p_mix_z1_ls, alpha_test_ls
    ):

        number_setting = number_split(
            p_pos_train_z0=combination[0],
            p_pos_train_z1=combination[1],
            p_mix_z1=combination[2],
            alpha_test=combination[3],
            train_test_ratio=train_test_ratio,
            n_test=n_test,
            verbose=False,
        )

        if number_setting is not None:

            #### Shorter Version!!!! Only select C_y == 0.5

            if round(number_setting["mix_param_dict"]["C_y"], 1) not in [0.5]:
                continue

            if round(number_setting["mix_param_dict"]["alpha_train"], 4) not in [
                1,
                1.5,
                0.6667,
                2,
                0.5,
                3,
                0.3333,
                4,
                0.25,
                6,
                0.1667,
            ]:
                continue

            if np.all(
                [number_setting[k] >= 10 for k in list(number_setting.keys())[:-1]]
            ):
                valid_full_settings.append(number_setting)
else:
    holdCy = False

    p_pos_test_z0_ls = np.arange(0.1, 1, 0.1)

    p_pos_train_z0_ls = np.arange(0.1, 1, 0.1)
    p_pos_train_z1_ls = np.arange(0.1, 1, 0.1)

    p_mix_z1_ls = np.arange(0, 1, 0.1)

    alpha_test_ls = np.concatenate(
        [np.float_power(10, np.linspace(start=-2, stop=2, num=50)), [0.2, 1, 5]]
    )

    p_pos_test_z1_ls = [
        i * j
        for i, j in itertools.product(p_pos_test_z0_ls, alpha_test_ls)
        if i * j <= 1
    ]

    valid_full_settings = []
    for combination in itertools.product(
        p_pos_train_z0_ls,
        p_pos_train_z1_ls,
        p_mix_z1_ls,
        [1],
        p_pos_test_z0_ls,
        p_pos_test_z1_ls,
    ):
        if combination[5] / combination[4] not in alpha_test_ls:
            continue

        if round(combination[1] / combination[0], 4) not in [1, 0.2, 5]:
            continue

        number_setting = number_split(
            p_pos_train_z0=combination[0],
            p_pos_train_z1=combination[1],
            p_mix_z1=combination[2],
            alpha_test=combination[3],
            train_test_ratio=train_test_ratio,
            n_test=n_test,
            verbose=False,
            holdCy=holdCy,
            p_pos_test_z0=combination[4],
            p_pos_test_z1=combination[5],
        )

        if number_setting is not None:
            if np.all(
                [number_setting[k] >= 10 for k in list(number_setting.keys())[:-1]]
            ):
                valid_full_settings.append(number_setting)


######## Model
#### Crazy version....

import warnings

warnings.simplefilter("ignore")

transform = args.transform

# transform = "Sentence-BERT"
# transform = "binaryUnigram"
# transform = "tfidf"

runs = 5

model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer = CountVectorizer(binary=True, min_df=1, stop_words="english")


### Hate Speech
if args.dataset == "HateSpeech":
    z_Categories = [
        "dynGen",
        "wsf",
    ]  # the order here matters! Should match with df0, df1
    label = "label_binary"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "dfSource"
    df0 = df_dynGen
    df1 = df_wsf

elif args.dataset == "SHAC":
    ## SHAC
    z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
    label = "Drug"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "location"
    df0 = df_shac_uw
    df1 = df_shac_mimic

elif args.dataset == "CD":
    z_Categories = ["avh", "r56"]
    label = "label_binary"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "dfSource"
    df0 = df_avh
    df1 = df_r56


### CivilComments from WILDS, by christian
# z_Categories = ["christian", "nonchristian"]  # the order here matters! Should match with df0, df1
# label='y_true'
# n_zCats = len(z_Categories)
# txt_col="text"
# domain_col = "Christian"
# df0 = df_christian
# df1 = df_nonchristian
# outdir = f"../output/regressionCivilComments_by_Christian"

### CivilComments from WILDS, by White
# z_Categories = ["white", "notwhite"]  # the order here matters! Should match with df0, df1
# label='y_true'
# n_zCats = len(z_Categories)
# txt_col="text"
# domain_col = "White"
# df0 = df_white
# df1 = df_notwhite
# outdir = f"../output/regressionCivilComments_by_White"

### CivilComments from WILDS, by Male
# z_Categories = ["male", "notmale"]  # the order here matters! Should match with df0, df1
# label='y_true'
# n_zCats = len(z_Categories)
# txt_col="text"
# domain_col = "Male"
# df0 = df_male
# df1 = df_notmale
# outdir = f"../output/regressionCivilComments_by_Male"

# save to file
# outname = f"../output/regressionInverseSHAC_MIMIC_UW/{transform}_{p_pos_train_z0}_{p_pos_train_z1}_{n_test}_{penalty}_C{C}_V{v}.pkl"
# outname = f"../output/regressionSHAC/{transform}_{p_pos_train_z0}_{p_pos_train_z1}_{n_test}_{penalty}_C{C}_V{v}.pkl"


### IMDB
# z_Categories = ["Horror","Documentary"]
# label='label_binary'
# n_zCats = len(z_Categories)
# z_Categories = ["Horror","nonHorror"]
# label='label_binary'
# n_zCats = len(z_Categories)

### Yelp by States, AZ vs MO
# z_Categories = ["AZ","MO"]
# label='label'
# n_zCats = len(z_Categories)
# txt_col = "text"
# domain_col = 'state'
# df0 = df_AZ
# df1 = df_MO

### Yelp by Year
# z_Categories = ["<=2015",">=2020"]
# label='label'
# n_zCats = len(z_Categories)
# txt_col = "text"
# domain_col = 'year_cut'
# df0 = df_before2015
# df1 = df_after2020


##### Test for LLaMa Average Embeddings
## SHAC

# # transform = "LLaMaAverageV2_7B"
# transform = "LLaMaAverageV2_13B"
# transform = "LLaMaAverageV2_70B_8Quant"

# runs = 5

# # SHAC
# z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
# label='Drug'
# n_zCats = len(z_Categories)
# txt_col="LLaMaEmbeddings"
# domain_col = "location"
# df0 = df_shac_llama_average_uw
# df1 = df_shac_llama_average_mimic
# outdir = f"../output/regressionSHACBalanceAlpha"

## Hate Speech

# transform = "LLaMaAverageV2_7B"
# transform = "LLaMaAverageV2_13B"
# transform = "LLaMaAverageV2_70B_8Quant"
# transform = "LLaMaAverageV2_7B_Permute"
# transform = "LLaMaAverageV2_13B_Permute"


# runs = 5

# z_Categories = ["dynGen", "wsf"]  # the order here matters! Should match with df0, df1
# label='label_binary'
# n_zCats = len(z_Categories)
# txt_col="LLaMaEmbeddings"
# domain_col = "dfSource"
# df0 = df_dynGen
# df1 = df_wsf
# outdir = f"../output/regressionHateSpeechBalanceAlpha"


outdir = f"../output/regression_ReSample_{args.dataset}_BalanceAlpha"
os.makedirs(outdir, exist_ok=True)


df0 = df0.assign(_id=lambda x: ["s0_" + str(x) for x in range(len(x))])
df1 = df1.assign(_id=lambda x: ["s1_" + str(x) for x in range(len(x))])

y_Categories = [0, 1]
n_yCats = len(y_Categories)


# setting for logistic regression
# penalty = "l1"
# solver = "liblinear"
penalty = "l2"
solver = "lbfgs"


random.seed(123)


for C, v in [
    [1, 10],
]:
    valid_n_full_settings = []

    retMetrics = {}

    for iRun in range(runs):

        _rand = random.randint(0, 2**32 - 1)
        print(_rand)

        print(iRun)
        for c in tqdm(
            valid_full_settings,
            file=open(
                f"../log/test_Aug_ReSample_{args.dataset}_ConstraintCy_{args.constraintCy}.txt",
                "w",
            ),
        ):
            # for c in test_settings:

            c = deepcopy(c)

            ###### Initial Split, to create testing set
            dfs = create_mix(
                df0=df0,
                df1=df1,
                target=label,
                setting=c,
                sample=False,
                # seed=random.randint(0,1000),
                seed=_rand,
            )

            if dfs is None:
                continue

            ##### Sampling for training set

            p_pos_train_target = max(
                c["mix_param_dict"]["p_pos_train_z0"],
                c["mix_param_dict"]["p_pos_train_z1"],
            )
            # p_pos_train_target=0.5

            c["run"] = iRun
            valid_n_full_settings.append(c)

            ##### Initial Embedding X (to be combined with mixup generations)
            #             if transform == "Sentence-BERT":
            #                 # use Sentence-BERT to encode sentences
            #                 x_transform_train = model.encode(dfs_new['train'][txt_col])
            #                 x_transform_test = model.encode(dfs['test'][txt_col])
            #             if transform == "binaryUnigram":
            #                 x_transform_train = vectorizer.fit_transform(dfs_new['train'][txt_col]).toarray()
            #                 x_transform_test = vectorizer.transform(dfs['test'][txt_col]).toarray()
            #             if transform in ["LLaMaAverage", "LLaMaAverageV2_7B", "LLaMaAverageV2_13B", "LLaMaAverageV2_70B_8Quant", "LLaMaAverageV2_7B_Permute", "LLaMaAverageV2_13B_Permute"]:
            #                 x_transform_train = np.stack(dfs_new['train'][txt_col])
            #                 x_transform_test = np.stack(dfs['test'][txt_col])
            #             if transform == "Clinical-BERT":
            #                 x_train_inputs = tokenizer(list(dfs_new['train'][txt_col]),
            #                                            return_tensors="pt", padding=True, truncation=True, max_length=256)
            #                 x_test_inputs = tokenizer(list(dfs['test'][txt_col]),
            #                                            return_tensors="pt", padding=True, truncation=True, max_length=256)

            #                 with torch.no_grad():
            #                     x_train_outputs = model(**x_train_inputs)
            #                     x_test_outputs = model(**x_test_inputs)

            #                 x_transform_train = x_train_outputs['last_hidden_state'][:,0,:]
            #                 x_transform_test = x_test_outputs['last_hidden_state'][:,0,:]

            #             # tfidf could be tricky...
            #             # https://stats.stackexchange.com/questions/154660/tfidfvectorizer-should-it-be-used-on-train-only-or-traintest
            #             if transform == "tfidf":
            #                 vectorizer = TfidfVectorizer(use_idf = True, ngram_range = (1,1))

            #                 # vectorizer.fit(pd.concat([dfs['train']['text'], dfs['test']['text']]))
            #                 vectorizer.fit(dfs_new['train']['text'])

            #                 x_transform_train = vectorizer.transform(dfs_new['train']['text']).toarray()
            #                 x_transform_test = vectorizer.transform(dfs['test']['text']).toarray()

            if transform == "binaryUnigram":
                dfs["train"]["embedding"] = list(
                    vectorizer.fit_transform(dfs["train"][txt_col]).toarray()
                )
                dfs["test"]["embedding"] = list(
                    vectorizer.transform(dfs["test"][txt_col]).toarray()
                )
            elif transform == "Sentence-BERT":
                # use Sentence-BERT to encode sentences
                dfs["train"]["embedding"] = list(model.encode(dfs["train"][txt_col]))
                dfs["test"]["embedding"] = list(model.encode(dfs["test"][txt_col]))

            ###### Split for Augmentation and Remove..
            dfs_train_original = deepcopy(dfs["train"])
            df0_train_pos = dfs["train"].query(
                f"(`{label}`==True) and ({domain_col} == '{z_Categories[0]}')"
            )
            df0_train_neg = dfs["train"].query(
                f"(`{label}`==False) and ({domain_col} == '{z_Categories[0]}')"
            )
            df1_train_pos = dfs["train"].query(
                f"(`{label}`==True) and ({domain_col} == '{z_Categories[1]}')"
            )
            df1_train_neg = dfs["train"].query(
                f"(`{label}`==False) and ({domain_col} == '{z_Categories[1]}')"
            )

            p_pos_train_target = max(
                c["mix_param_dict"]["p_pos_train_z0"],
                c["mix_param_dict"]["p_pos_train_z1"],
            )

            for _t in np.arange(0, 1, 0.01):
                c_train_target = number_split(
                    p_pos_train_z0=p_pos_train_target,
                    p_pos_train_z1=p_pos_train_target,
                    p_mix_z1=c["mix_param_dict"]["p_mix_z1"],
                    alpha_test=_t,  # c['mix_param_dict']['alpha_test'],
                    train_test_ratio=train_test_ratio,
                    n_test=int(n_test),
                    verbose=False,
                )

            def augDF(df, name_to_aug):
                n_augs = c_train_target[name_to_aug] - c[name_to_aug]

                if n_augs < 0:
                    df = df.sample(
                        n=c_train_target[name_to_aug],
                        random_state=_rand,
                        replace=False,
                        ignore_index=True,
                    )
                elif n_augs > 0:

                    _df = df.sample(
                        n=n_augs,
                        random_state=_rand,
                        replace=True,
                        ignore_index=True,
                    )

                    df = pd.concat([df, _df], ignore_index=True).reset_index(drop=True)

                return df

            df0_train_pos = augDF(df0_train_pos, "n_z0_pos_train")
            df0_train_neg = augDF(df0_train_neg, "n_z0_neg_train")
            df1_train_pos = augDF(df1_train_pos, "n_z1_pos_train")
            df1_train_neg = augDF(df1_train_neg, "n_z1_neg_train")

            dfs["train"] = pd.concat(
                [df0_train_pos, df0_train_neg, df1_train_pos, df1_train_neg],
                ignore_index=True,
            ).reset_index(drop=True)

            ####### Add a few columns in c
            c["mix_param_dict"]["p_pos_train_z0_new"] = p_pos_train_target
            c["mix_param_dict"]["p_pos_train_z1_new"] = p_pos_train_target
            c["mix_param_dict"]["C_y_train"] = (
                c["mix_param_dict"]["p_pos_train_z0_new"]
                * c["mix_param_dict"]["p_mix_z0"]
                + c["mix_param_dict"]["p_pos_train_z1_new"]
                * c["mix_param_dict"]["p_mix_z1"]
            )

            x_transform_train = dfs["train"]["embedding"].apply(pd.Series).to_numpy()
            x_transform_test = dfs["test"]["embedding"].apply(pd.Series).to_numpy()

            ### Initiate Y (to be combined with mixup generations)
            y_train = dfs["train"][label]
            y_test = dfs["test"][label]

            n_test = len(y_test)

            df_test = dfs["test"]
            z_train = dfs["train"][domain_col]

            n_train = len(dfs["train"])

            assert n_test * train_test_ratio == n_train == len(y_train) == len(z_train)

            ############## Before Training...

            # calculate P(Z): NOTE: this may not be useful, because it is pre-defined!!!! (to be combined with mixup generations)
            p_z = []

            for i in z_Categories:
                p_z.append(sum(z_train == i) / (n_train))

            confounders_train = (
                pd.get_dummies(
                    pd.Categorical(z_train, categories=z_Categories),
                    prefix="confounder",
                )
                * v
            )
            confounders_test = (
                pd.get_dummies(
                    pd.Categorical(dfs["test"][domain_col], categories=z_Categories),
                    prefix="confounder",
                )
                * v
            )

            #####################  Confound: statistical Adjustment
            clf = LogisticRegression(
                penalty=penalty, C=C, max_iter=1000, class_weight=None, solver=solver
            )

            # for training set: add confounder as dummy variables
            x_embeddings_confound_train = np.concatenate(
                [x_transform_train, confounders_train], axis=1
            )

            # for testing set: construct pseudo-confounders, and then add them as dummy variables into embeddings
            x_embeddings_confound_test_ls = []

            for i in range(n_zCats):
                a = np.empty((n_test, n_zCats))
                a.fill(0)
                a[:, i] = v
                _ = np.concatenate([x_transform_test, a], axis=1)
                x_embeddings_confound_test_ls.append(_)

            # fit the training data, add confounding dummy variables as predictors
            clf.fit(X=x_embeddings_confound_train, y=y_train)

            # prediction on testing data: use pseudo-confounders, store predictions for all scenarios of Z
            y_probs_ls = []

            for i in range(n_zCats):
                _y_probs = clf.predict_proba(X=x_embeddings_confound_test_ls[i])
                y_probs_ls.append(_y_probs)

            # calculate P(Y|X): sum(P(y|x,z) * P(z))
            y_probs_confound = np.empty((n_test, n_yCats))
            y_probs_confound.fill(0)

            for i in range(n_zCats):
                y_probs_confound += y_probs_ls[i] * p_z[i]

            idx_df0 = dfs["test"][domain_col] == z_Categories[0]
            idx_df1 = dfs["test"][domain_col] == z_Categories[1]

            _ = appendMetrics(
                ret=retMetrics,
                sufix="backdoor",
                y_true=y_test,
                y_prob=y_probs_confound[:, 1],
                f1_cutoff=0.5,
            )

            _ = appendMetrics(
                ret=retMetrics,
                sufix="backdoor_df0",
                y_true=y_test[idx_df0],
                y_prob=y_probs_confound[idx_df0, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="backdoor_df1",
                y_true=y_test[idx_df1],
                y_prob=y_probs_confound[idx_df1, 1],
                f1_cutoff=0.5,
            )

            #####################  Simple Logistic Regression, WITHOUT confounder
            clf_vanilla = LogisticRegression(
                penalty=penalty, C=C, max_iter=1000, class_weight=None, solver=solver
            )

            clf_vanilla.fit(X=x_transform_train, y=y_train)

            y_probs_vanilla = clf_vanilla.predict_proba(X=x_transform_test)

            _ = appendMetrics(
                ret=retMetrics,
                sufix="vanilla",
                y_true=y_test,
                y_prob=y_probs_vanilla[:, 1],
                f1_cutoff=0.5,
            )

            _ = appendMetrics(
                ret=retMetrics,
                sufix="vanilla_df0",
                y_true=y_test[idx_df0],
                y_prob=y_probs_vanilla[idx_df0, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="vanilla_df1",
                y_true=y_test[idx_df1],
                y_prob=y_probs_vanilla[idx_df1, 1],
                f1_cutoff=0.5,
            )

    ############  Put Results in DataFrame, with extra information (a little redundant)

    # organize results in DataFrame
    df_eval = pd.DataFrame(retMetrics)

    for k in valid_n_full_settings[0]["mix_param_dict"].keys():
        df_eval[k] = [_dict["mix_param_dict"][k] for _dict in valid_n_full_settings]

    for k in valid_n_full_settings[0].keys():
        if k != "mix_param_dict":
            df_eval[k] = [_dict[k] for _dict in valid_n_full_settings]

    if args.constraintCy:
        outname = f"{outdir}/ConstraintCy-AugUP-ReSample-{transform}-ntest_{n_test}-{penalty}-C{C}-V{v}.pkl"
    else:
        outname = f"{outdir}/UnconstraintCy-AugUP-ReSample-{transform}-ntest_{n_test}-{penalty}-C{C}-V{v}.pkl"

    with open(outname, "wb") as f:
        pickle.dump(df_eval, file=f)
