import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import sys

sys.path.append("../src")
sys.path.append("../config")

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
from sklearn.linear_model import LinearRegression
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
from sklearn.calibration import calibration_curve

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
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT

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
parser.add_argument("-c", "--CombinationIdx", type=int, help="Set idx of c to use")
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
parser.add_argument(
    "--useIW",
    action="store_true",
    help="whether to use IW",
)
args = parser.parse_args()


def confusion_matrix_probs(y_true, y_pred, y_0_label=0):

    idx = y_true == y_0_label
    n_t0_p0 = sum(1 - y_pred[:, 1][idx])
    n_t0_p1 = sum(y_pred[:, 1][idx])

    n_t1_p0 = sum(1 - y_pred[:, 1][~idx])
    n_t1_p1 = sum(y_pred[:, 1][~idx])

    return np.array([[n_t0_p0, n_t1_p0], [n_t0_p1, n_t1_p1]])


def emUpdate(pt_x_w1_ls, pw1_0, pt_w1):

    pt_w0 = 1 - pt_w1
    pw1_s = pw1_0

    i_em = 0
    ct = 0
    while 1:
        pw1_s_pre = pw1_s
        pw0_s = 1 - pw1_s

        pw1_x_s_ls = []

        for pt_x_w1 in pt_x_w1_ls:
            pt_x_w0 = 1 - pt_x_w1

            deno = pw0_s / pt_w0 * pt_x_w0 + pw1_s / pt_w1 * pt_x_w1
            pw1_x_s = pw1_s / pt_w1 * pt_x_w1 / deno
            pw1_x_s_ls.append(pw1_x_s)

        pw1_s = np.mean(pw1_x_s_ls)
        i_em += 1

        if abs(pw1_s_pre - pw1_s) < 0.0001:
            ct += 1
            if ct == 20:
                break
        # print("iter: " + str(i_em))

    return pw1_x_s_ls, pw1_s


pick_C = args.CombinationIdx

######## Load Data
if args.dataset == "SHAC":
    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x["Drug"] else 0, axis=1)
    df_shac["dfSource"] = df_shac["location"]

    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

    n_test = 200

    p_pos_train_z0_ls = SHAC_DICT["Run-0"]["p_pos_train_z0_ls"]
    p_pos_train_z1_ls = SHAC_DICT["Run-0"]["p_pos_train_z1_ls"]
    p_mix_z1_ls = SHAC_DICT["Run-0"]["p_mix_z1_ls"]
    c = SHAC_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_2800"

elif args.dataset == "HateSpeech":
    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()

    # n_test = 1000
    n_test = 200

    p_pos_train_z0_ls = HateSpeech_DICT["Run-2"]["p_pos_train_z0_ls"]
    p_pos_train_z1_ls = HateSpeech_DICT["Run-2"]["p_pos_train_z1_ls"]
    p_mix_z1_ls = HateSpeech_DICT["Run-2"]["p_mix_z1_ls"]
    c = HateSpeech_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n1000_9870"

elif args.dataset == "CD":
    df_all = load_cd()
    df_avh = df_all["avh"]
    df_r56 = df_all["r56"]

    n_test = 200

    p_pos_train_z0_ls = HateSpeech_DICT["Run-1"]["p_pos_train_z0_ls"]
    p_pos_train_z1_ls = HateSpeech_DICT["Run-1"]["p_pos_train_z1_ls"]
    p_mix_z1_ls = HateSpeech_DICT["Run-1"]["p_mix_z1_ls"]
    c = CD_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_566"


else:
    sys.exit("no such dataset for processing")


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


##### Fix Training Set

print("Balanced? Check setting....")
print(c)
dfs_train = create_mix(
    df0=df0,
    df1=df1,
    target=label,
    setting=c,
    sample=False,
    # seed=random.randint(0,1000),
    seed=222,
)


####### Diff Out Dataset used in LoRA Fine-Tuning

assert dfs_train is not None

df0 = df0[~df0[txt_col].isin(dfs_train["train"][txt_col])].reset_index(drop=True)
df0 = df0[~df0[txt_col].isin(dfs_train["test"][txt_col])].reset_index(drop=True)


df1 = df1[~df1[txt_col].isin(dfs_train["train"][txt_col])].reset_index(drop=True)
df1 = df1[~df1[txt_col].isin(dfs_train["test"][txt_col])].reset_index(drop=True)


############  Get Split Configs & Further Limit Sampling Set, if necessary
train_test_ratio = 4


alpha_test_ls = np.concatenate(
    [
        np.float_power(
            10,
            np.linspace(
                start=-2,
                stop=2,
                num=40,  # TODO
                # num=11,
            ),
        ),
        [0.2, 1, 5],
    ]
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
        if np.all([number_setting[k] >= 10 for k in list(number_setting.keys())[:-1]]):
            valid_full_settings.append(number_setting)


valid_n_full_settings = []

for c in tqdm(valid_full_settings):
    c = c.copy()
    c["n_train"] = 0
    c["n_z0_pos_train"] = 1
    c["n_z0_neg_train"] = 1
    c["n_z1_pos_train"] = 1
    c["n_z1_neg_train"] = 1
    c["mix_param_dict"]["p_pos_train_z0"] = 0
    c["mix_param_dict"]["p_pos_train_z1"] = 0
    c["mix_param_dict"]["p_pos_train"] = 0
    c["mix_param_dict"]["alpha_train"] = 0

    # create train/test split according to stats
    dfs = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, seed=222)

    if dfs is None:
        continue

    valid_n_full_settings.append(c)

tmp_df = [x["mix_param_dict"] for x in valid_n_full_settings]

tmp_df = pd.DataFrame(tmp_df)

tmp_df["alpha_train"] = tmp_df["alpha_train"].round(4)
tmp_df["C_y1"] = np.floor(tmp_df["C_y"] * 10) / 10
tmp_df["combination"] = valid_n_full_settings

valid_full_settings = tmp_df["combination"]


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


if args.clf == "LR":
    lr_name = "regression"
elif args.clf == "SVM":
    lr_name = "SVM_"

outdir = f"../output/{lr_name}{args.dataset}BalanceAlpha"
# if not args.constraintCy:
#     outdir = f"../output/{lr_name}{args.dataset}BalanceAlpha_UnlimitCy"

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

## TODO:
# setting for logistic regression
penalty = "l1"
solver = "liblinear"
# penalty = "l2"
# solver = "lbfgs"


random.seed(123)


valid_n_full_settings = []

# [[1,10], [1,1],[1,100]]


########---------     PreCalculation

if transform == "Sentence-BERT":
    # use Sentence-BERT to encode sentences
    x_transform_train = model.encode(dfs_train["train"][txt_col])
if transform == "binaryUnigram":
    x_transform_train = vectorizer.fit_transform(dfs_train["train"][txt_col]).toarray()
if transform in [
    "LLaMaAverage",
    "LLaMaAverageV2_7B",
    "LLaMaAverageV2_13B",
    "LLaMaAverageV2_70B_8Quant",
    "LLaMaAverageV2_7B_Permute",
    "LLaMaAverageV2_13B_Permute",
]:
    x_transform_train = np.stack(dfs_train["train"][txt_col])

# tfidf could be tricky...
# https://stats.stackexchange.com/questions/154660/tfidfvectorizer-should-it-be-used-on-train-only-or-traintest
if transform == "tfidf":
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 1))

    # vectorizer.fit(pd.concat([dfs['train']['text'], dfs['test']['text']]))
    vectorizer.fit(dfs_train["train"]["text"])

    x_transform_train = vectorizer.transform(dfs_train["train"]["text"]).toarray()


for C, v in [
    [1, 10],
]:

    retMetrics = {}

    valid_n_full_settings = []

    for iRun in range(runs):

        _rand = random.randint(0, 2**32 - 1)
        print(_rand)

        print(iRun)
        for c in tqdm(
            valid_full_settings, file=open(f"../log/iw_{args.dataset}.txt", "w")
        ):
            c = c.copy()

            # create train/test split according to stats

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
            # valid_n_full_settings.append(c)

            if transform == "Sentence-BERT":
                # use Sentence-BERT to encode sentences
                # x_transform_train = model.encode(dfs_train["train"][txt_col])
                x_transform_test = model.encode(dfs["test"][txt_col])
            if transform == "binaryUnigram":
                # x_transform_train = vectorizer.fit_transform(
                #     dfs_train["train"][txt_col]
                # ).toarray()
                x_transform_test = vectorizer.transform(dfs["test"][txt_col]).toarray()
            if transform in [
                "LLaMaAverage",
                "LLaMaAverageV2_7B",
                "LLaMaAverageV2_13B",
                "LLaMaAverageV2_70B_8Quant",
                "LLaMaAverageV2_7B_Permute",
                "LLaMaAverageV2_13B_Permute",
            ]:
                # x_transform_train = np.stack(dfs_train["train"][txt_col])
                x_transform_test = np.stack(dfs["test"][txt_col])

            # tfidf could be tricky...
            # https://stats.stackexchange.com/questions/154660/tfidfvectorizer-should-it-be-used-on-train-only-or-traintest
            if transform == "tfidf":
                # vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 1))

                # # vectorizer.fit(pd.concat([dfs['train']['text'], dfs['test']['text']]))
                # vectorizer.fit(dfs_train["train"]["text"])

                # x_transform_train = vectorizer.transform(
                #     dfs_train["train"]["text"]
                # ).toarray()
                x_transform_test = vectorizer.transform(dfs["test"]["text"]).toarray()

            # tfidf could be tricky...
            # https://stats.stackexchange.com/questions/154660/tfidfvectorizer-should-it-be-used-on-train-only-or-traintest
            #     elif transform == "tfidf":
            #         vectorizer = TfidfVectorizer(use_idf = True, ngram_range = (1,1))

            #         vectorizer.fit(pd.concat([dfs['train']['text'], dfs['test']['text']]))

            #         x_transform_train = vectorizer.transform(dfs['train']['text']).toarray()
            #         x_transform_test = vectorizer.transform(dfs['test']['text']).toarray()

            y_train = dfs_train["train"][label]
            y_test = dfs["test"][label]

            z_train = dfs_train["train"][domain_col]
            z_test = dfs["test"][domain_col]

            ## For binary case
            z_train = np.array([0 if z == z_Categories[0] else 1 for z in z_train])
            z_test = np.array([0 if z == z_Categories[0] else 1 for z in z_test])

            n_test = len(y_test)

            df_test = dfs["test"]

            # calculate P(Z): NOTE: this may not be useful, because it is pre-defined!!!!
            p_z = []

            for i in z_Categories:
                p_z.append(
                    sum(dfs_train["train"][domain_col] == i) / len(dfs_train["train"])
                )

            confounders_train = (
                pd.get_dummies(
                    pd.Categorical(
                        dfs_train["train"][domain_col], categories=z_Categories
                    ),
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
                p_z_pred_ls = []
                p_z_pred_EM_ls = []

                kf = KFold(n_splits=crossfitSplit, shuffle=True, random_state=271)
                for idx_train, idx_test in kf.split(x_transform_train):
                    # 0 set is the smaller set; 1 set is the larger set
                    x_transform_train_0 = x_transform_train[idx_test]
                    x_transform_train_1 = x_transform_train[idx_train]
                    z_train_0 = z_train[idx_test]
                    z_train_1 = z_train[idx_train]

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

                    f0.fit(x_transform_train_0, z_train_0)

                    # use second half to get Confusion Matrix c
                    z_train_1_pred = f0.predict_proba(x_transform_train_1)
                    # C_mat = confusion_matrix(
                    #     y_true=z_train_1, y_pred=z_train_1_pred, labels=y_Categories
                    # ).T
                    C_mat = confusion_matrix_probs(
                        y_true=z_train_1, y_pred=z_train_1_pred
                    )
                    z_train_1_freq = []
                    for _ in range(2):
                        z_train_1_freq.append(sum(z_train_1 == _))
                    z_train_1_freq = np.array(z_train_1_freq)

                    C_mat = C_mat / z_train_1_freq

                    # predict Pq(fx) on test set
                    z_test_f0 = f0.predict(x_transform_test)
                    z_test_f0_margin = np.zeros(shape=(len(z_Categories), 1))

                    z_test_f0_margin = [
                        sum(z_test_f0 == x) / len(z_test_f0) for x in range(2)
                    ]
                    z_test_f0_margin = np.expand_dims(z_test_f0_margin, -1)

                    # get weight w
                    # eigenvalues, eigenvectors = np.linalg.eig(C_mat)
                    # if min(eigenvalues) <= hyperparam_delta:
                    #     w = np.ones((len(z_Categories), 1))
                    # else:
                    #     w = np.clip(
                    #         np.matmul(np.linalg.inv(C_mat), z_test_f0_margin),
                    #         a_min=0,
                    #         a_max=None,
                    #     )

                    # build final predictor f
                    ### NOTE: Use IW
                    if args.useIW:
                        ## NOTE: do not do clip, even when probability < 0 ... for now
                        w = np.matmul(np.linalg.inv(C_mat), z_test_f0_margin)
                        cw_dict = {
                            _y: w[_idx, 0] for _idx, _y in enumerate(z_Categories)
                        }
                        cw = [w[_idx, 0] for _idx, _y in enumerate(z_Categories)]
                    else:
                        ### NOTE: Do NOT Use IW. Use Prediction directly
                        cw = z_test_f0_margin.squeeze()

                    ### NOTE: EM
                    z_test_f0_prob = f0.predict_proba(x_transform_test)[:, 1]
                    tmp_p_z = [sum(z_train_0 == x) / len(z_train_0) for x in range(2)]

                    cw_EM = emUpdate(
                        pt_x_w1_ls=z_test_f0_prob,
                        pw1_0=tmp_p_z[1],
                        pt_w1=tmp_p_z[1],
                    )

                    p_z_pred_ls.append(cw)
                    p_z_pred_EM_ls.append(np.array([1 - cw_EM[1], cw_EM[1]]))

                p_z_pred = np.mean(p_z_pred_ls, axis=0)
                p_z_pred_EM = np.mean(p_z_pred_EM_ls, axis=0)

            elif crossfitSplit == 0:
                ################## Importance Weighting

                x_transform_train_0 = x_transform_train[idx_0]
                x_transform_train_1 = x_transform_train[idx_1]
                z_train_0 = z_train[idx_0]
                z_train_1 = z_train[idx_1]

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

                f0.fit(x_transform_train_0, z_train_0)

                # use second half to get Confusion Matrix c
                z_train_1_pred = f0.predict_proba(x_transform_train_1)
                # C_mat = confusion_matrix(
                #     y_true=z_train_1, y_pred=z_train_1_pred, labels=y_Categories
                # ).T
                C_mat = confusion_matrix_probs(y_true=z_train_1, y_pred=z_train_1_pred)
                z_train_1_freq = []
                for _ in range(2):
                    z_train_1_freq.append(sum(z_train_1 == _))
                z_train_1_freq = np.array(z_train_1_freq)

                C_mat = C_mat / z_train_1_freq

                # predict Pq(fx) on test set
                z_test_f0 = f0.predict(x_transform_test)
                z_test_f0_margin = np.zeros(shape=(len(z_Categories), 1))

                z_test_f0_margin = [
                    sum(z_test_f0 == x) / len(z_test_f0) for x in range(2)
                ]
                z_test_f0_margin = np.expand_dims(z_test_f0_margin, -1)

                # get weight w
                # eigenvalues, eigenvectors = np.linalg.eig(C_mat)
                # if min(eigenvalues) <= hyperparam_delta:
                #     w = np.ones((len(z_Categories), 1))
                # else:
                #     w = np.clip(
                #         np.matmul(np.linalg.inv(C_mat), z_test_f0_margin),
                #         a_min=0,
                #         a_max=None,
                #     )
                ## NOTE: do not do clip, even when probability < 0 ... for now
                w = np.matmul(np.linalg.inv(C_mat), z_test_f0_margin)

                # build final predictor f
                ### NOTE: Use IW
                if args.useIW:
                    cw_dict = {_y: w[_idx, 0] for _idx, _y in enumerate(z_Categories)}
                    cw = [w[_idx, 0] for _idx, _y in enumerate(z_Categories)]
                else:
                    ### NOTE: Do NOT Use IW
                    cw = z_test_f0_margin.squeeze()

                ## NOTE: EM
                z_test_f0_prob = f0.predict_proba(x_transform_test)[:, 1]
                tmp_p_z = [sum(z_train_0 == x) / len(z_train_0) for x in range(2)]

                cw_EM = emUpdate(
                    pt_x_w1_ls=z_test_f0_prob,
                    pw1_0=tmp_p_z[1],
                    pt_w1=tmp_p_z[1],
                )

                p_z_pred = np.array(cw)
                p_z_pred_EM = np.array([1 - cw_EM[1], cw_EM[1]])

            p_z_true_1 = c["mix_param_dict"]["C_z"]
            p_z_true = [1 - p_z_true_1, p_z_true_1]

            c["p_z_pred_IW_1"] = p_z_pred[1]
            c["p_z_simpleInfer_1"] = p_z[1]
            c["p_z_pred_EM_1"] = p_z_pred_EM[1]
            valid_n_full_settings.append(c)

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

            # calculate P(Y|X): sum(P(y|x,z) * P(z))
            y_probs_confound = np.empty((n_test, n_yCats))
            y_probs_confound.fill(0)

            for i in range(n_zCats):
                y_probs_confound += y_probs_ls[i] * p_z[i]

            ### Adjusted Z - pred

            # calculate P(Y|X): sum(P(y|x,z) * P(z))
            y_probs_IW = np.empty((n_test, n_yCats))
            y_probs_IW.fill(0)

            for i in range(n_zCats):
                y_probs_IW += y_probs_ls[i] * p_z_pred[i]

            ### Adjusted Z - true Z
            # calculate P(Y|X): sum(P(y|x,z) * P(z))
            y_probs_IWTrueZ = np.empty((n_test, n_yCats))
            y_probs_IWTrueZ.fill(0)

            for i in range(n_zCats):
                y_probs_IWTrueZ += y_probs_ls[i] * p_z_true[i]

            ### EM
            y_probs_EM = np.empty((n_test, n_yCats))
            y_probs_EM.fill(0)

            for i in range(n_zCats):
                y_probs_EM += y_probs_ls[i] * p_z_pred_EM[i]

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

            # #####################  Simple Logistic Regression, WITHOUT confounder, but naive loss balancing
            # clf_naive_lossBalance = LogisticRegression(
            #     penalty=penalty,
            #     C=C,
            #     max_iter=1000,
            #     class_weight="balanced",
            #     solver=solver,
            # )

            # clf_naive_lossBalance.fit(X=x_transform_train, y=y_train)

            # y_probs_naive_lossBalance = clf_naive_lossBalance.predict_proba(
            #     X=x_transform_test
            # )

            idx_df0 = df_test[domain_col] == z_Categories[0]
            idx_df1 = df_test[domain_col] == z_Categories[1]

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

            _ = appendMetrics(
                ret=retMetrics,
                sufix="confound",
                y_true=y_test,
                y_prob=y_probs_confound[:, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="confound_df0",
                y_true=y_test[idx_df0],
                y_prob=y_probs_confound[idx_df0, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="confound_df1",
                y_true=y_test[idx_df1],
                y_prob=y_probs_confound[idx_df1, 1],
                f1_cutoff=0.5,
            )

            _ = appendMetrics(
                ret=retMetrics,
                sufix="IW",
                y_true=y_test,
                y_prob=y_probs_IW[:, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="IW_df0",
                y_true=y_test[idx_df0],
                y_prob=y_probs_IW[idx_df0, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="IW_df1",
                y_true=y_test[idx_df1],
                y_prob=y_probs_IW[idx_df1, 1],
                f1_cutoff=0.5,
            )

            _ = appendMetrics(
                ret=retMetrics,
                sufix="EM",
                y_true=y_test,
                y_prob=y_probs_EM[:, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="EM_df0",
                y_true=y_test[idx_df0],
                y_prob=y_probs_EM[idx_df0, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="EM_df1",
                y_true=y_test[idx_df1],
                y_prob=y_probs_EM[idx_df1, 1],
                f1_cutoff=0.5,
            )

            _ = appendMetrics(
                ret=retMetrics,
                sufix="TrueZ",
                y_true=y_test,
                y_prob=y_probs_IWTrueZ[:, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="TrueZ_df0",
                y_true=y_test[idx_df0],
                y_prob=y_probs_IWTrueZ[idx_df0, 1],
                f1_cutoff=0.5,
            )
            _ = appendMetrics(
                ret=retMetrics,
                sufix="TrueZ_df1",
                y_true=y_test[idx_df1],
                y_prob=y_probs_IWTrueZ[idx_df1, 1],
                f1_cutoff=0.5,
            )

            # auprc_logistic_confounder_df0.append(
            #     metrics.average_precision_score(
            #         y_true=y_test[df_test[domain_col] == z_Categories[0]],
            #         y_score=y_probs_confound[df_test[domain_col] == z_Categories[0], 1],
            #     )
            # )
            # auprc_logistic_confounder_df1.append(
            #     metrics.average_precision_score(
            #         y_true=y_test[df_test[domain_col] == z_Categories[1]],
            #         y_score=y_probs_confound[df_test[domain_col] == z_Categories[1], 1],
            #     )
            # )
            # auprc_logistic_IW_df0.append(
            #     metrics.average_precision_score(
            #         y_true=y_test[df_test[domain_col] == z_Categories[0]],
            #         y_score=y_probs_IW[df_test[domain_col] == z_Categories[0], 1],
            #     )
            # )
            # auprc_logistic_IW_df1.append(
            #     metrics.average_precision_score(
            #         y_true=y_test[df_test[domain_col] == z_Categories[1]],
            #         y_score=y_probs_IW[df_test[domain_col] == z_Categories[1], 1],
            #     )
            # )
            # auprc_logistic_EM_df0.append(
            #     metrics.average_precision_score(
            #         y_true=y_test[df_test[domain_col] == z_Categories[0]],
            #         y_score=y_probs_EM[df_test[domain_col] == z_Categories[0], 1],
            #     )
            # )
            # auprc_logistic_EM_df1.append(
            #     metrics.average_precision_score(
            #         y_true=y_test[df_test[domain_col] == z_Categories[1]],
            #         y_score=y_probs_EM[df_test[domain_col] == z_Categories[1], 1],
            #     )
            # )
            # auroc_logistic_confounder.append(
            #     roc_auc_score(y_true=y_test, y_score=y_probs_confound[:, 1])
            # )
            # t_confounder = precision_recall_fscore_support(
            #     y_true=y_test,
            #     y_pred=y_probs_confound[:, 1] > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # t_con_df0 = precision_recall_fscore_support(
            #     y_true=y_test[df_test[domain_col] == z_Categories[0]],
            #     y_pred=y_probs_confound[df_test[domain_col] == z_Categories[0], 1]
            #     > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # t_con_df1 = precision_recall_fscore_support(
            #     y_true=y_test[df_test[domain_col] == z_Categories[1]],
            #     y_pred=y_probs_confound[df_test[domain_col] == z_Categories[1], 1]
            #     > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # precision_confounder.append(t_confounder[0])
            # recall_confounder.append(t_confounder[1])
            # f1_confounder.append(t_confounder[2])
            # precision_confounder_df0.append(t_con_df0[0])
            # recall_confounder_df0.append(t_con_df0[1])
            # f1_confounder_df0.append(t_con_df0[2])
            # precision_confounder_df1.append(t_con_df1[0])
            # recall_confounder_df1.append(t_con_df1[1])
            # f1_confounder_df1.append(t_con_df1[2])

            # auprc_logistic_vanilla.append(
            #     metrics.average_precision_score(
            #         y_true=y_test, y_score=y_probs_vanilla[:, 1]
            #     )
            # )
            # auprc_logistic_vanilla_df0.append(
            #     metrics.average_precision_score(
            #         y_true=y_test[df_test[domain_col] == z_Categories[0]],
            #         y_score=y_probs_vanilla[df_test[domain_col] == z_Categories[0], 1],
            #     )
            # )
            # auprc_logistic_vanilla_df1.append(
            #     metrics.average_precision_score(
            #         y_true=y_test[df_test[domain_col] == z_Categories[1]],
            #         y_score=y_probs_vanilla[df_test[domain_col] == z_Categories[1], 1],
            #     )
            # )
            # auroc_logistic_vanilla.append(
            #     roc_auc_score(y_true=y_test, y_score=y_probs_vanilla[:, 1])
            # )
            # t_vanilla = precision_recall_fscore_support(
            #     y_true=y_test,
            #     y_pred=y_probs_vanilla[:, 1] > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # t_van_df0 = precision_recall_fscore_support(
            #     y_true=y_test[df_test[domain_col] == z_Categories[0]],
            #     y_pred=y_probs_vanilla[df_test[domain_col] == z_Categories[0], 1] > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # t_van_df1 = precision_recall_fscore_support(
            #     y_true=y_test[df_test[domain_col] == z_Categories[1]],
            #     y_pred=y_probs_vanilla[df_test[domain_col] == z_Categories[1], 1] > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # precision_vanilla.append(t_vanilla[0])
            # recall_vanilla.append(t_vanilla[1])
            # f1_vanilla.append(t_vanilla[2])
            # precision_vanilla_df0.append(t_van_df0[0])
            # recall_vanilla_df0.append(t_van_df0[1])
            # f1_vanilla_df0.append(t_van_df0[2])
            # precision_vanilla_df1.append(t_van_df1[0])
            # recall_vanilla_df1.append(t_van_df1[1])
            # f1_vanilla_df1.append(t_van_df1[2])

            # t_IW = precision_recall_fscore_support(
            #     y_true=y_test,
            #     y_pred=y_probs_IW[:, 1] > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # f1_IW.append(t_IW[2])
            # auprc_logistic_IW.append(
            #     metrics.average_precision_score(y_true=y_test, y_score=y_probs_IW[:, 1])
            # )

            # # auprc_logistic_naiveLossBalance.append(
            # #     metrics.average_precision_score(
            # #         y_true=y_test, y_score=y_probs_naive_lossBalance[:, 1]
            # #     )
            # # )

            # auroc_logistic_IW.append(
            #     roc_auc_score(y_true=y_test, y_score=y_probs_IW[:, 1])
            # )

            # t_IWTrueZ = precision_recall_fscore_support(
            #     y_true=y_test,
            #     y_pred=y_probs_IWTrueZ[:, 1] > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # f1_IWTrueZ.append(t_IWTrueZ[2])
            # auprc_logistic_IWTrueZ.append(
            #     metrics.average_precision_score(
            #         y_true=y_test, y_score=y_probs_IWTrueZ[:, 1]
            #     )
            # )

            # auroc_logistic_IWTrueZ.append(
            #     roc_auc_score(y_true=y_test, y_score=y_probs_IWTrueZ[:, 1])
            # )

            # # auroc_logistic_naiveLossBalance.append(
            # #     roc_auc_score(y_true=y_test, y_score=y_probs_naive_lossBalance[:, 1])
            # # )

            # calibration_vanilla.append(
            #     getCalibrationSlope(
            #         y_true=y_test, y_prob=y_probs_vanilla[:, 1], n_bins=10
            #     )
            # )
            # calibration_confounder.append(
            #     getCalibrationSlope(
            #         y_true=y_test, y_prob=y_probs_confound[:, 1], n_bins=10
            #     )
            # )
            # calibration_IW.append(
            #     getCalibrationSlope(y_true=y_test, y_prob=y_probs_IW[:, 1], n_bins=10)
            # )
            # calibration_IWTrueZ.append(
            #     getCalibrationSlope(
            #         y_true=y_test, y_prob=y_probs_IWTrueZ[:, 1], n_bins=10
            #     )
            # )
            # calibration_EM.append(
            #     getCalibrationSlope(y_true=y_test, y_prob=y_probs_EM[:, 1], n_bins=10)
            # )

            # t_EM = precision_recall_fscore_support(
            #     y_true=y_test,
            #     y_pred=y_probs_EM[:, 1] > 0.5,
            #     average="binary",
            #     pos_label=1,
            # )
            # f1_EM.append(t_EM[2])
            # auprc_logistic_EM.append(
            #     metrics.average_precision_score(y_true=y_test, y_score=y_probs_EM[:, 1])
            # )

            # auroc_logistic_EM.append(
            #     roc_auc_score(y_true=y_test, y_score=y_probs_EM[:, 1])
            # )

    ############  Put Results in DataFrame, with extra information (a little redundant)

    # organize results in DataFrame
    df_eval = pd.DataFrame(retMetrics)

    # df_eval = pd.DataFrame(
    #     {
    #         "auprc_logistic_confounder": auprc_logistic_confounder,
    #         "auprc_logistic_vanilla": auprc_logistic_vanilla,
    #         "auprc_logistic_confounder_df0": auprc_logistic_confounder_df0,
    #         "auprc_logistic_confounder_df1": auprc_logistic_confounder_df1,
    #         "precision_confounder": precision_confounder,
    #         "recall_confounder": recall_confounder,
    #         "f1_confounder": f1_confounder,
    #         "precision_confounder_df0": precision_confounder_df0,
    #         "recall_confounder_df0": recall_confounder_df0,
    #         "f1_confounder_df0": f1_confounder_df0,
    #         "precision_confounder_df1": precision_confounder_df1,
    #         "recall_confounder_df1": recall_confounder_df1,
    #         "f1_confounder_df1": f1_confounder_df1,
    #         "auprc_logistic_vanilla_df0": auprc_logistic_vanilla_df0,
    #         "auprc_logistic_vanilla_df1": auprc_logistic_vanilla_df1,
    #         "precision_vanilla": precision_vanilla,
    #         "recall_vanilla": recall_vanilla,
    #         "f1_vanilla": f1_vanilla,
    #         "precision_vanilla_df0": precision_vanilla_df0,
    #         "recall_vanilla_df0": recall_vanilla_df0,
    #         "f1_vanilla_df0": f1_vanilla_df0,
    #         "precision_vanilla_df1": precision_vanilla_df1,
    #         "recall_vanilla_df1": recall_vanilla_df1,
    #         "f1_vanilla_df1": f1_vanilla_df1,
    #         "auprc_logistic_IW": auprc_logistic_IW,
    #         "auprc_logistic_IWTrueZ": auprc_logistic_IWTrueZ,
    #         "f1_IW": f1_IW,
    #         "f1_IWTrueZ": f1_IWTrueZ,
    #         # "auprc_logistic_naiveLossBalance": auprc_logistic_naiveLossBalance,
    #         "auroc_logistic_IW": auroc_logistic_IW,
    #         "auroc_logistic_IWTrueZ": auroc_logistic_IWTrueZ,
    #         # "auroc_logistic_naiveLossBalance": auroc_logistic_naiveLossBalance,
    #         "auroc_logistic_vanilla": auroc_logistic_vanilla,
    #         "auroc_logistic_confounder": auroc_logistic_confounder,
    #         "calibration_EM": calibration_EM,
    #         "auprc_logistic_EM": auprc_logistic_EM,
    #         "f1_EM": f1_EM,
    #         "auroc_logistic_EM": auroc_logistic_EM,
    #         "calibration_vanilla": calibration_vanilla,
    #         "calibration_confounder": calibration_confounder,
    #         "calibration_IW": calibration_IW,
    #         "calibration_IWTrueZ": calibration_IWTrueZ,
    #     }
    # )

    for k in valid_n_full_settings[0]["mix_param_dict"].keys():
        df_eval[k] = [_dict["mix_param_dict"][k] for _dict in valid_n_full_settings]

    for k in valid_n_full_settings[0].keys():
        if k != "mix_param_dict":
            df_eval[k] = [_dict[k] for _dict in valid_n_full_settings]

    # outname = f"{outdir}/{transform}_{p_pos_train_z0}_{p_pos_train_z1}_{n_test}_{penalty}_C{C}_V{v}.pkl"
    if args.useIW:
        pre_deco = "Importance_Weighting"
    else:
        pre_deco = "NoIW"
    if crossfitSplit != 0:
        outname = f"{outdir}/{pre_deco}_forZ_Combination-{pick_C}_Double-HyperDelta_{hyperparam_delta}-CrossFit_{crossfitSplit}-{transform}-ntest_{n_test}-{penalty}_C{C}-V{v}.pkl"
    else:
        outname = f"{outdir}/{pre_deco}_forZ_Combination-{pick_C}_Double-HyperDelta_{hyperparam_delta}-{transform}-ntest_{n_test}-{penalty}_C{C}-V{v}.pkl"

    with open(outname, "wb") as f:
        pickle.dump(df_eval, file=f)
