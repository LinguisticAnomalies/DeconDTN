import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

from utils import number_split, create_mix
from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC
from custom_distance import KL
from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_CD import load_cd

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
    "--crossfitFolds",
    type=int,
    default=5,
    help="crossfit folds. If 0, no cross fitting is used.",
)
parser.add_argument(
    "--iwDelta",
    type=float,
    default=0.00001,
    help="default delta value in IW for determining smallest eigenvalue of C",
)
parser.add_argument(
    "--clf",
    type=str,
    default="LR",
    choices=["LR", "SVM"],
    help="classifier type",
)
args = parser.parse_args()


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

    p_pos_train_z0_ls = np.arange(0.1, 1, 0.1)
    p_pos_train_z1_ls = np.arange(0.1, 1, 0.1)

    p_pos_test_z0_ls = np.arange(0.1, 1, 0.1)

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

        if round(combination[1] / combination[0], 4) not in [
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
crossfitSplit = args.crossfitFolds
hyperparam_delta = args.iwDelta

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

if args.clf == "LR":
    lr_name = "regression"
elif args.clf == "SVM":
    lr_name = "SVM_"

outdir = f"../output/{lr_name}{args.dataset}BalanceAlpha"
if not args.constraintCy:
    outdir = f"../output/{lr_name}{args.dataset}BalanceAlpha_UnlimitCy"

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


os.makedirs(outdir, exist_ok=True)


y_Categories = [0, 1]
n_yCats = len(y_Categories)


# setting for logistic regression
# penalty = "l1"
# solver = "liblinear"
penalty = "l2"
solver = "lbfgs"


random.seed(123)
auprc_logistic_confounder = []
auprc_logistic_confounder_df0 = []
auprc_logistic_confounder_df1 = []

auprc_logistic_vanilla = []
auprc_logistic_vanilla_df0 = []
auprc_logistic_vanilla_df1 = []

valid_n_full_settings = []

# [[1,10], [1,1],[1,100]]
for C, v in [
    [1, 10],
]:
    auprc_logistic_confounder = []
    auprc_logistic_confounder_df0 = []
    auprc_logistic_confounder_df1 = []
    auroc_logistic_confounder = []
    precision_confounder = []
    recall_confounder = []
    f1_confounder = []
    precision_confounder_df0 = []
    recall_confounder_df0 = []
    f1_confounder_df0 = []
    precision_confounder_df1 = []
    recall_confounder_df1 = []
    f1_confounder_df1 = []

    auprc_logistic_vanilla = []
    auprc_logistic_vanilla_df0 = []
    auprc_logistic_vanilla_df1 = []
    precision_vanilla = []
    recall_vanilla = []
    f1_vanilla = []
    precision_vanilla_df0 = []
    recall_vanilla_df0 = []
    f1_vanilla_df0 = []
    precision_vanilla_df1 = []
    recall_vanilla_df1 = []
    f1_vanilla_df1 = []

    auprc_logistic_IW = []
    f1_IW = []

    auprc_logistic_naiveLossBalance = []
    auroc_logistic_IW = []
    auroc_logistic_naiveLossBalance = []
    auroc_logistic_vanilla = []

    valid_n_full_settings = []

    for iRun in range(runs):

        _rand = random.randint(0, 2**32 - 1)
        print(_rand)

        print(iRun)
        for c in tqdm(
            valid_full_settings, file=open(f"../log/iw_{args.dataset}.txt", "w")
        ):
            # for c in test_settings:
            # if round(c["mix_param_dict"]["C_y"],1) not in [0.5]:
            #     print(c)
            #     sys.exit()
            c = c.copy()

            # create train/test split according to stats
            # dfs = create_mix(df0=df_wls_merge, df1=df_adress, target='label', setting= c, sample=False)
            # dfs = create_mix(df0=df_shac_uw, df1=df_shac_mimic, target=label, setting= c, sample=False, seed=random.randint(0,1000))
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

            # #### TO DELETE: For results on Selected C_y ONLY!!!!!!!!!
            # if round(c['mix_param_dict']['C_y'], 4) not in [0.36, 0.44, 0.52, 0.24, 0.54, 0.84]:
            #     continue

            c["run"] = iRun
            valid_n_full_settings.append(c)

            if transform == "Sentence-BERT":
                # use Sentence-BERT to encode sentences
                x_transform_train = model.encode(dfs["train"][txt_col])
                x_transform_test = model.encode(dfs["test"][txt_col])
            if transform == "binaryUnigram":
                x_transform_train = vectorizer.fit_transform(
                    dfs["train"][txt_col]
                ).toarray()
                x_transform_test = vectorizer.transform(dfs["test"][txt_col]).toarray()
            if transform in [
                "LLaMaAverage",
                "LLaMaAverageV2_7B",
                "LLaMaAverageV2_13B",
                "LLaMaAverageV2_70B_8Quant",
                "LLaMaAverageV2_7B_Permute",
                "LLaMaAverageV2_13B_Permute",
            ]:
                x_transform_train = np.stack(dfs["train"][txt_col])
                x_transform_test = np.stack(dfs["test"][txt_col])

            # tfidf could be tricky...
            # https://stats.stackexchange.com/questions/154660/tfidfvectorizer-should-it-be-used-on-train-only-or-traintest
            if transform == "tfidf":
                vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 1))

                # vectorizer.fit(pd.concat([dfs['train']['text'], dfs['test']['text']]))
                vectorizer.fit(dfs["train"]["text"])

                x_transform_train = vectorizer.transform(dfs["train"]["text"]).toarray()
                x_transform_test = vectorizer.transform(dfs["test"]["text"]).toarray()

            # tfidf could be tricky...
            # https://stats.stackexchange.com/questions/154660/tfidfvectorizer-should-it-be-used-on-train-only-or-traintest
            #     elif transform == "tfidf":
            #         vectorizer = TfidfVectorizer(use_idf = True, ngram_range = (1,1))

            #         vectorizer.fit(pd.concat([dfs['train']['text'], dfs['test']['text']]))

            #         x_transform_train = vectorizer.transform(dfs['train']['text']).toarray()
            #         x_transform_test = vectorizer.transform(dfs['test']['text']).toarray()

            y_train = dfs["train"][label]
            y_test = dfs["test"][label]

            n_test = len(y_test)

            df_test = dfs["test"]

            confounders_train = (
                pd.get_dummies(
                    pd.Categorical(dfs["train"][domain_col], categories=z_Categories),
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

            idx_0 = np.random.choice(
                range(len(y_train)), int(len(y_train) * 0.5), replace=False
            )
            idx_1 = np.array(list(set(range(len(y_train))) - set(idx_0)))

            ################## Importance Weighting

            if crossfitSplit != 0:
                y_probs_IW_kfold = []
                kf = KFold(n_splits=crossfitSplit, shuffle=True, random_state=271)
                for idx_train, idx_test in kf.split(x_transform_train):
                    # 0 set is the smaller set; 1 set is the larger set
                    x_transform_train_0 = x_transform_train[idx_test]
                    x_transform_train_1 = x_transform_train[idx_train]
                    y_train_0 = y_train[idx_test]
                    y_train_1 = y_train[idx_train]

                    # use 1st half to train classifier
                    if args.clf == "LR":
                        f0 = LogisticRegression(
                            penalty=penalty,
                            C=C,
                            max_iter=1000,
                            solver=solver,
                            class_weight=None,
                        )
                    elif args.clf == "SVM":
                        f0 = make_pipeline(
                            StandardScaler(),
                            SVC(
                                probability=True,
                                gamma="scale",
                                C=C,
                                random_state=42,
                                class_weight=None,
                            ),
                        )

                    f0.fit(x_transform_train_0, y_train_0)

                    # use second half to get Confusion Matrix c
                    y_train_1_pred = f0.predict(x_transform_train_1)
                    # C_mat = confusion_matrix(
                    #     y_true=y_train_1, y_pred=y_train_1_pred, labels=y_Categories
                    # ).T
                    C_mat = confusion_matrix_probs(
                        y_true=y_train_1, y_pred=y_train_1_pred
                    ).T
                    C_mat = C_mat / len(y_train_1_pred)

                    # predict Pq(fx) on test set
                    y_test_f0 = f0.predict(x_transform_test)
                    y_test_f0_margin = np.zeros(shape=(len(y_Categories), 1))

                    y_test_f0_margin = [
                        sum(y_test_f0 == x) / len(y_test_f0) for x in y_Categories
                    ]
                    y_test_f0_margin = np.expand_dims(y_test_f0_margin, -1)

                    # get weight w
                    eigenvalues, eigenvectors = np.linalg.eig(C_mat)
                    if min(eigenvalues) <= hyperparam_delta:
                        w = np.ones((len(y_Categories), 1))
                    else:
                        w = np.clip(
                            np.matmul(np.linalg.inv(C_mat), y_test_f0_margin),
                            a_min=0,
                            a_max=None,
                        )

                    # build final predictor f
                    cw_dict = {_y: w[_idx, 0] for _idx, _y in enumerate(y_Categories)}
                    if args.clf == "LR":
                        f = LogisticRegression(
                            penalty=penalty,
                            C=C,
                            max_iter=1000,
                            solver=solver,
                            class_weight=cw_dict,
                        )
                    elif args.clf == "SVM":
                        f = make_pipeline(
                            StandardScaler(),
                            SVC(
                                probability=True,
                                gamma="scale",
                                C=C,
                                random_state=42,
                                class_weight=cw_dict,
                            ),
                        )

                    f.fit(x_transform_train_0, y_train_0)

                    y_probs_IW_kfold.append(f.predict_proba(X=x_transform_test))

                y_probs_IW = np.mean(y_probs_IW_kfold, axis=0)
            elif crossfitSplit == 0:
                ################## Importance Weighting

                x_transform_train_0 = x_transform_train[idx_0]
                x_transform_train_1 = x_transform_train[idx_1]
                y_train_0 = y_train[idx_0]
                y_train_1 = y_train[idx_1]

                # use 1st half to train classifier
                if args.clf == "LR":
                    f0 = LogisticRegression(
                        penalty=penalty,
                        C=C,
                        max_iter=1000,
                        solver=solver,
                        class_weight=None,
                    )
                elif args.clf == "SVM":
                    f0 = make_pipeline(
                        StandardScaler(),
                        SVC(
                            probability=True,
                            gamma="scale",
                            C=C,
                            random_state=42,
                            class_weight=None,
                        ),
                    )

                f0.fit(x_transform_train_0, y_train_0)

                # use second half to get Confusion Matrix c
                y_train_1_pred = f0.predict(x_transform_train_1)
                # C_mat = confusion_matrix(
                #     y_true=y_train_1, y_pred=y_train_1_pred, labels=y_Categories
                # ).T
                C_mat = confusion_matrix_probs(
                    y_true=y_train_1, y_pred=y_train_1_pred
                ).T
                C_mat = C_mat / len(y_train_1_pred)

                # predict Pq(fx) on test set
                y_test_f0 = f0.predict(x_transform_test)
                y_test_f0_margin = np.zeros(shape=(len(y_Categories), 1))

                y_test_f0_margin = [
                    sum(y_test_f0 == x) / len(y_test_f0) for x in y_Categories
                ]
                y_test_f0_margin = np.expand_dims(y_test_f0_margin, -1)

                # get weight w
                eigenvalues, eigenvectors = np.linalg.eig(C_mat)
                if min(eigenvalues) <= hyperparam_delta:
                    w = np.ones((len(y_Categories), 1))
                else:
                    w = np.clip(
                        np.matmul(np.linalg.inv(C_mat), y_test_f0_margin),
                        a_min=0,
                        a_max=None,
                    )

                # build final predictor f
                cw_dict = {_y: w[_idx, 0] for _idx, _y in enumerate(y_Categories)}
                if args.clf == "LR":
                    f = LogisticRegression(
                        penalty=penalty,
                        C=C,
                        max_iter=1000,
                        solver=solver,
                        class_weight=cw_dict,
                    )
                elif args.clf == "SVM":
                    f = make_pipeline(
                        StandardScaler(),
                        SVC(
                            probability=True,
                            gamma="scale",
                            C=C,
                            random_state=42,
                            class_weight=cw_dict,
                        ),
                    )

                f.fit(x_transform_train_0, y_train_0)

                y_probs_IW = f.predict_proba(X=x_transform_test)

            #####################  Confound: statistical Adjustment
            if args.clf == "LR":
                clf = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    max_iter=1000,
                    class_weight=None,
                    solver=solver,
                )
            elif args.clf == "SVM":
                clf = make_pipeline(
                    StandardScaler(),
                    SVC(probability=True, gamma="scale", C=C, random_state=42),
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

            # calculate P(Z): NOTE: this may not be useful, because it is pre-defined!!!!
            p_z = []

            for i in z_Categories:
                p_z.append(sum(dfs["train"][domain_col] == i) / len(dfs["train"]))

            # calculate P(Y|X): sum(P(y|x,z) * P(z))
            y_probs_confound = np.empty((n_test, n_yCats))
            y_probs_confound.fill(0)

            for i in range(n_zCats):
                y_probs_confound += y_probs_ls[i] * p_z[i]

            auprc_logistic_confounder.append(
                metrics.average_precision_score(
                    y_true=y_test, y_score=y_probs_confound[:, 1]
                )
            )
            auprc_logistic_confounder_df0.append(
                metrics.average_precision_score(
                    y_true=y_test[df_test[domain_col] == z_Categories[0]],
                    y_score=y_probs_confound[df_test[domain_col] == z_Categories[0], 1],
                )
            )
            auprc_logistic_confounder_df1.append(
                metrics.average_precision_score(
                    y_true=y_test[df_test[domain_col] == z_Categories[1]],
                    y_score=y_probs_confound[df_test[domain_col] == z_Categories[1], 1],
                )
            )
            auroc_logistic_confounder.append(
                roc_auc_score(y_true=y_test, y_score=y_probs_confound[:, 1])
            )
            t_confounder = precision_recall_fscore_support(
                y_true=y_test,
                y_pred=y_probs_confound[:, 1] > 0.5,
                average="binary",
                pos_label=1,
            )
            t_con_df0 = precision_recall_fscore_support(
                y_true=y_test[df_test[domain_col] == z_Categories[0]],
                y_pred=y_probs_confound[df_test[domain_col] == z_Categories[0], 1]
                > 0.5,
                average="binary",
                pos_label=1,
            )
            t_con_df1 = precision_recall_fscore_support(
                y_true=y_test[df_test[domain_col] == z_Categories[1]],
                y_pred=y_probs_confound[df_test[domain_col] == z_Categories[1], 1]
                > 0.5,
                average="binary",
                pos_label=1,
            )
            precision_confounder.append(t_confounder[0])
            recall_confounder.append(t_confounder[1])
            f1_confounder.append(t_confounder[2])
            precision_confounder_df0.append(t_con_df0[0])
            recall_confounder_df0.append(t_con_df0[1])
            f1_confounder_df0.append(t_con_df0[2])
            precision_confounder_df1.append(t_con_df1[0])
            recall_confounder_df1.append(t_con_df1[1])
            f1_confounder_df1.append(t_con_df1[2])

            #####################  Simple Logistic Regression, WITHOUT confounder
            if args.clf == "LR":
                clf_vanilla = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    max_iter=1000,
                    class_weight=None,
                    solver=solver,
                )
            elif args.clf == "SVM":
                clf_vanilla = make_pipeline(
                    StandardScaler(),
                    SVC(probability=True, gamma="scale", C=C, random_state=42),
                )

            clf_vanilla.fit(X=x_transform_train, y=y_train)

            y_probs_vanilla = clf_vanilla.predict_proba(X=x_transform_test)

            #####################  Simple Logistic Regression, WITHOUT confounder, but naive loss balancing
            clf_naive_lossBalance = LogisticRegression(
                penalty=penalty,
                C=C,
                max_iter=1000,
                class_weight="balanced",
                solver=solver,
            )

            clf_naive_lossBalance.fit(X=x_transform_train, y=y_train)

            y_probs_naive_lossBalance = clf_naive_lossBalance.predict_proba(
                X=x_transform_test
            )

            auprc_logistic_vanilla.append(
                metrics.average_precision_score(
                    y_true=y_test, y_score=y_probs_vanilla[:, 1]
                )
            )
            auprc_logistic_vanilla_df0.append(
                metrics.average_precision_score(
                    y_true=y_test[df_test[domain_col] == z_Categories[0]],
                    y_score=y_probs_vanilla[df_test[domain_col] == z_Categories[0], 1],
                )
            )
            auprc_logistic_vanilla_df1.append(
                metrics.average_precision_score(
                    y_true=y_test[df_test[domain_col] == z_Categories[1]],
                    y_score=y_probs_vanilla[df_test[domain_col] == z_Categories[1], 1],
                )
            )
            auroc_logistic_vanilla.append(
                roc_auc_score(y_true=y_test, y_score=y_probs_vanilla[:, 1])
            )
            t_vanilla = precision_recall_fscore_support(
                y_true=y_test,
                y_pred=y_probs_vanilla[:, 1] > 0.5,
                average="binary",
                pos_label=1,
            )
            t_van_df0 = precision_recall_fscore_support(
                y_true=y_test[df_test[domain_col] == z_Categories[0]],
                y_pred=y_probs_vanilla[df_test[domain_col] == z_Categories[0], 1] > 0.5,
                average="binary",
                pos_label=1,
            )
            t_van_df1 = precision_recall_fscore_support(
                y_true=y_test[df_test[domain_col] == z_Categories[1]],
                y_pred=y_probs_vanilla[df_test[domain_col] == z_Categories[1], 1] > 0.5,
                average="binary",
                pos_label=1,
            )
            precision_vanilla.append(t_vanilla[0])
            recall_vanilla.append(t_vanilla[1])
            f1_vanilla.append(t_vanilla[2])
            precision_vanilla_df0.append(t_van_df0[0])
            recall_vanilla_df0.append(t_van_df0[1])
            f1_vanilla_df0.append(t_van_df0[2])
            precision_vanilla_df1.append(t_van_df1[0])
            recall_vanilla_df1.append(t_van_df1[1])
            f1_vanilla_df1.append(t_van_df1[2])

            t_IW = precision_recall_fscore_support(
                y_true=y_test,
                y_pred=y_probs_IW[:, 1] > 0.5,
                average="binary",
                pos_label=1,
            )
            f1_IW.append(t_IW[2])
            auprc_logistic_IW.append(
                metrics.average_precision_score(y_true=y_test, y_score=y_probs_IW[:, 1])
            )

            auprc_logistic_naiveLossBalance.append(
                metrics.average_precision_score(
                    y_true=y_test, y_score=y_probs_naive_lossBalance[:, 1]
                )
            )

            auroc_logistic_IW.append(
                roc_auc_score(y_true=y_test, y_score=y_probs_IW[:, 1])
            )

            auroc_logistic_naiveLossBalance.append(
                roc_auc_score(y_true=y_test, y_score=y_probs_naive_lossBalance[:, 1])
            )

    ############  Put Results in DataFrame, with extra information (a little redundant)

    # organize results in DataFrame
    df_eval = pd.DataFrame(
        {
            "auprc_logistic_confounder": auprc_logistic_confounder,
            "auprc_logistic_vanilla": auprc_logistic_vanilla,
            "auprc_logistic_confounder_df0": auprc_logistic_confounder_df0,
            "auprc_logistic_confounder_df1": auprc_logistic_confounder_df1,
            "precision_confounder": precision_confounder,
            "recall_confounder": recall_confounder,
            "f1_confounder": f1_confounder,
            "precision_confounder_df0": precision_confounder_df0,
            "recall_confounder_df0": recall_confounder_df0,
            "f1_confounder_df0": f1_confounder_df0,
            "precision_confounder_df1": precision_confounder_df1,
            "recall_confounder_df1": recall_confounder_df1,
            "f1_confounder_df1": f1_confounder_df1,
            "auprc_logistic_vanilla_df0": auprc_logistic_vanilla_df0,
            "auprc_logistic_vanilla_df1": auprc_logistic_vanilla_df1,
            "precision_vanilla": precision_vanilla,
            "recall_vanilla": recall_vanilla,
            "f1_vanilla": f1_vanilla,
            "precision_vanilla_df0": precision_vanilla_df0,
            "recall_vanilla_df0": recall_vanilla_df0,
            "f1_vanilla_df0": f1_vanilla_df0,
            "precision_vanilla_df1": precision_vanilla_df1,
            "recall_vanilla_df1": recall_vanilla_df1,
            "f1_vanilla_df1": f1_vanilla_df1,
            "auprc_logistic_IW": auprc_logistic_IW,
            "f1_IW": f1_IW,
            "auprc_logistic_naiveLossBalance": auprc_logistic_naiveLossBalance,
            "auroc_logistic_IW": auroc_logistic_IW,
            "auroc_logistic_naiveLossBalance": auroc_logistic_naiveLossBalance,
            "auroc_logistic_vanilla": auroc_logistic_vanilla,
            "auroc_logistic_confounder": auroc_logistic_confounder,
        }
    )

    for k in valid_n_full_settings[0]["mix_param_dict"].keys():
        df_eval[k] = [_dict["mix_param_dict"][k] for _dict in valid_n_full_settings]

    for k in valid_n_full_settings[0].keys():
        if k != "mix_param_dict":
            df_eval[k] = [_dict[k] for _dict in valid_n_full_settings]

    # outname = f"{outdir}/{transform}_{p_pos_train_z0}_{p_pos_train_z1}_{n_test}_{penalty}_C{C}_V{v}.pkl"
    if crossfitSplit != 0:
        outname = f"{outdir}/Importance_Weighting-HyperDelta_{hyperparam_delta}-CrossFit_{crossfitSplit}-{transform}-ntest_{n_test}-{penalty}_C{C}-V{v}.pkl"
    else:
        outname = f"{outdir}/Importance_Weighting-HyperDelta_{hyperparam_delta}-{transform}-ntest_{n_test}-{penalty}_C{C}-V{v}.pkl"

    with open(outname, "wb") as f:
        pickle.dump(df_eval, file=f)
