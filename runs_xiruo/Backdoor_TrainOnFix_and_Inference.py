import os
import argparse

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="SHAC",
    help="Dataset for the experiment",
)
parser.add_argument("-c", "--CombinationIdx", type=int, help="Set idx of c to use")
parser.add_argument(
    "--model_name", default="roberta-base", help="Model to use. Default RoBERTa"
)
parser.add_argument(
    "--nTest",
    type=int,
    default=200,
    help="Number of testing samples",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Specify cuda GPU",
)
parser.add_argument("--output_dir", type=str, help="Directory to save outputs for backdoor")
parser.add_argument(
    "--output_dir_noBackdoor", type=str, help="Directory to save outputs for no backdoor"
)
parser.add_argument(
    "--mntdir",
    type=str,
    default="/bime-munin/",
    help="Number of testing samples",
)
parser.add_argument(
    "--transform",
    type=str,
    default="binaryUnigram",
    choices=["binaryUnigram", "Sentence-BERT"],
    help="Tranform text into vectors",
)
parser.add_argument(
    "--runs",
    type=int,
    default=1,
    help="Iterations to run",
)

## To Delete??


parser.add_argument("--weightsEdited", type=str, help="Path to edited weights")
parser.add_argument(
    "--sampleValidSettings",
    action="store_true",
    help="whether to sample valid settings",
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for eval dataset"
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="On which GPU to run",
)

parser.add_argument(
    "--cpuOps",
    action="store_true",
    help="Unload to CPU for tensors. This still stores state_dict() on GPU in the end",
)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import random
import sys

sys.path.append("../src")
sys.path.append("../config")

sys.path.append("../src")
sys.path.append("../config")

from utils import create_mix

from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_SHAC import load_process_SHAC
from process_CD import load_cd
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT
import warnings
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainerCallback,
    default_data_collator,
)
from scipy.special import softmax
from accelerate.utils import load_and_quantize_model

# from accelerate.utils import BnbQuantizationConfig


# ===================================================
pick_C = args.CombinationIdx
n_test = args.nTest


class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.model_name = args.model_name
globalconfig.max_seq_length = 512
globalconfig.device = args.device


######  Load Data

### SHAC
if args.dataset == "SHAC":
    z_category = ["uw", "mimic"]
    y_Categories = ["False", "True"]
    txt_col = "text"
    domain_col = "location"
elif args.dataset == "HateSpeech":
    z_category = ["dynGen", "wsf"]
    y_Categories = [0, 1]
    txt_col = "text"
    domain_col = "dfSource"
elif args.dataset == "CD":
    z_category = ["avh", "r56"]
    y_Categories = [0, 1]
    txt_col = "text"
    domain_col = "dfSource"


output_dir = args.output_dir

if args.dataset == "SHAC":
    label = "Drug"

    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x[label] else 0, axis=1)
    df_shac["dfSource"] = df_shac[domain_col]

    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

    df0 = df_shac_uw
    df1 = df_shac_mimic
    df_split_label = "Drug"

    c = SHAC_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_2800"

    z_category = ["uw", "mimic"]
    y_Categories = ["False", "True"]
    txt_col = "text"
    domain_col = "location"

elif args.dataset == "HateSpeech":
    label = "label"
    ## Hate Speech data already have "label_binary" and dfSource
    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()

    df0 = df_dynGen
    df1 = df_wsf
    df_split_label = "label_binary"

    c = HateSpeech_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n1000_9870"

    z_category = ["dynGen", "wsf"]
    y_Categories = [0, 1]
    txt_col = "text"
    domain_col = "dfSource"

elif args.dataset == "CD":
    label = "label"
    df_all = load_cd()
    df_avh = df_all["avh"]
    df_r56 = df_all["r56"]

    df0 = df_avh
    df1 = df_r56
    df_split_label = "label_binary"

    c = CD_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_566"

    z_category = ["avh", "r56"]
    y_Categories = [0, 1]
    txt_col = "text"
    domain_col = "dfSource"


# output_dir = f"../output/regressionSHACBalanceAlpha"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(args.output_dir_noBackdoor, exist_ok=True)

## Training
print("Balanced? Check setting....")
print(c)
dfs = create_mix(
    df0=df0,
    df1=df1,
    target=df_split_label,
    setting=c,
    sample=False,
    # seed=random.randint(0,1000),
    seed=222,
)


warnings.simplefilter("ignore")


def doInference(
    df_in,
    x_transform,
    clf_backdoor,
    clf_noBackdoor,
    p_z,
    z_category,
    v,
    n_yCats,
):
    confounders_test = (
        pd.get_dummies(
            pd.Categorical(df_in[domain_col], categories=z_category),
            prefix="confounder",
        )
        * v
    )

    n_test_actual = len(df_in)
    n_zCats = len(z_category)

    # for testing set: construct pseudo-confounders, and then add them as dummy variables into embeddings
    x_embeddings_confound_test_ls = []

    for i in range(n_zCats):
        a = np.empty((n_test_actual, n_zCats))
        a.fill(0)
        a[:, i] = v
        _ = np.concatenate([x_transform, a], axis=1)
        x_embeddings_confound_test_ls.append(_)

    # prediction on testing data: use pseudo-confounders, store predictions for all scenarios of Z
    y_probs_ls = []

    for i in range(n_zCats):
        _y_probs = clf_backdoor.predict_proba(X=x_embeddings_confound_test_ls[i])
        y_probs_ls.append(_y_probs)

    # calculate P(Y|X): sum(P(y|x,z) * P(z))
    y_probs_confound = np.empty((n_test_actual, n_yCats))
    y_probs_confound.fill(0)

    for i in range(n_zCats):
        y_probs_confound += y_probs_ls[i] * p_z[i]

    y_probs_vanilla = clf_noBackdoor.predict_proba(X=x_transform)

    return {
        "probs_backdoor": y_probs_confound,
        "probs_noBackdoor": y_probs_vanilla,
    }


transform = args.transform

runs = args.runs

model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer = CountVectorizer(binary=True, min_df=1, stop_words="english")


n_yCats = len(y_Categories)
n_zCats = len(z_category)

# setting for logistic regression

penalty = "l2"
solver = "lbfgs"


random.seed(123)
# valid_n_full_settings = []

# [[1,10], [1,1],[1,100]]
for C, v in [
    [1, 10],
]:

    for iRun in range(runs):

        _rand = random.randint(0, 2**32 - 1)
        print(_rand)

        print(iRun)

        if transform == "Sentence-BERT":
            # use Sentence-BERT to encode sentences
            x_transform_train = model.encode(dfs["train"][txt_col])

            # x_transform_test = model.encode(dfs["test"][txt_col])

            x_transform_test_df0 = model.encode(df0[txt_col])
            x_transform_test_df1 = model.encode(df1[txt_col])

        if transform == "binaryUnigram":
            x_transform_train = vectorizer.fit_transform(
                dfs["train"][txt_col]
            ).toarray()

            # x_transform_test = vectorizer.transform(dfs["test"][txt_col]).toarray()

            x_transform_test_df0 = vectorizer.transform(df0[txt_col]).toarray()
            x_transform_test_df1 = vectorizer.transform(df1[txt_col]).toarray()

        # if transform in ["LLaMaAverage", "LLaMaAverageV2_7B", "LLaMaAverageV2_13B", "LLaMaAverageV2_70B_8Quant", "LLaMaAverageV2_7B_Permute", "LLaMaAverageV2_13B_Permute"]:
        #     x_transform_train = np.stack(dfs['train'][txt_col])
        #     x_transform_test = np.stack(dfs['test'][txt_col])
        # if transform == "Clinical-BERT":
        #     x_train_inputs = tokenizer(list(dfs['train'][txt_col]),
        #                                 return_tensors="pt", padding=True, truncation=True, max_length=256)
        #     x_test_inputs = tokenizer(list(dfs['test'][txt_col]),
        #                                 return_tensors="pt", padding=True, truncation=True, max_length=256)

        #     with torch.no_grad():
        #         x_train_outputs = model(**x_train_inputs)
        #         x_test_outputs = model(**x_test_inputs)

        #     x_transform_train = x_train_outputs['last_hidden_state'][:,0,:]
        #     x_transform_test = x_test_outputs['last_hidden_state'][:,0,:]

        y_train = dfs["train"][label]
        # y_test = dfs["test"][label]

        # n_test_actual = len(y_test)

        # df_test = dfs["test"]

        confounders_train = (
            pd.get_dummies(
                pd.Categorical(dfs["train"][domain_col], categories=z_category),
                prefix="confounder",
            )
            * v
        )

        # confounders_test = (
        #     pd.get_dummies(
        #         pd.Categorical(dfs["test"][domain_col], categories=z_category),
        #         prefix="confounder",
        #     )
        #     * v
        # )

        #####################  Confound: Backdoor Adjustment
        clf = LogisticRegression(
            penalty=penalty, C=C, max_iter=1000, class_weight=None, solver=solver
        )

        # for training set: add confounder as dummy variables
        x_embeddings_confound_train = np.concatenate(
            [x_transform_train, confounders_train], axis=1
        )

        # fit the training data, add confounding dummy variables as predictors
        clf.fit(X=x_embeddings_confound_train, y=y_train)

        # calculate P(Z): NOTE: this may not be useful, because it is pre-defined!!!!
        p_z = []

        for i in z_category:
            p_z.append(sum(dfs["train"][domain_col] == i) / len(dfs["train"]))

        #####################  Simple Logistic Regression, WITHOUT confounder
        clf_vanilla = LogisticRegression(
            penalty=penalty, C=C, max_iter=1000, class_weight=None, solver=solver
        )

        clf_vanilla.fit(X=x_transform_train, y=y_train)

        ##################### Inference

        inference_df0 = doInference(
            df_in=df0,
            x_transform=x_transform_test_df0,
            clf_backdoor=clf,
            clf_noBackdoor=clf_vanilla,
            p_z=p_z,
            z_category=z_category,
            v=v,
            n_yCats=n_yCats,
        )

        inference_df1 = doInference(
            df_in=df1,
            x_transform=x_transform_test_df1,
            clf_backdoor=clf,
            clf_noBackdoor=clf_vanilla,
            p_z=p_z,
            z_category=z_category,
            v=v,
            n_yCats=n_yCats,
        )

        df0_backdoor = df0.copy()
        df0_vanilla = df0.copy()

        df1_backdoor = df1.copy()
        df1_vanilla = df1.copy()

        df0_backdoor[["ycat_0", "ycat_1"]] = inference_df0["probs_backdoor"]
        df0_vanilla[["ycat_0", "ycat_1"]] = inference_df0["probs_noBackdoor"]

        df1_backdoor[["ycat_0", "ycat_1"]] = inference_df1["probs_backdoor"]
        df1_vanilla[["ycat_0", "ycat_1"]] = inference_df1["probs_noBackdoor"]

        ######  Load Data & Save
        if args.dataset == "SHAC":

            df0_backdoor.to_csv(
                f"{output_dir}/inference_set-{pick_C}-irun-{iRun}_df_shac_uw.csv"
            )
            df1_backdoor.to_csv(
                f"{output_dir}/inference_set-{pick_C}-irun-{iRun}_df_shac_mimic.csv"
            )

            df0_vanilla.to_csv(
                f"{args.output_dir_noBackdoor}/inference_set-{pick_C}-irun-{iRun}_df_shac_uw.csv"
            )
            df1_vanilla.to_csv(
                f"{args.output_dir_noBackdoor}/inference_set-{pick_C}-irun-{iRun}_df_shac_mimic.csv"
            )

        elif args.dataset == "HateSpeech":
            df0_backdoor.to_csv(
                f"{output_dir}/inference_set-{pick_C}-irun-{iRun}_df_dynGen.csv"
            )
            df1_backdoor.to_csv(
                f"{output_dir}/inference_set-{pick_C}-irun-{iRun}_df_wsf.csv"
            )

            df0_vanilla.to_csv(
                f"{args.output_dir_noBackdoor}/inference_set-{pick_C}-irun-{iRun}_df_dynGen.csv"
            )
            df1_vanilla.to_csv(
                f"{args.output_dir_noBackdoor}/inference_set-{pick_C}-irun-{iRun}_df_wsf.csv"
            )

        elif args.dataset == "CD":

            df0_backdoor.to_csv(
                f"{output_dir}/inference_set-{pick_C}-irun-{iRun}_df_avh.csv"
            )
            df1_backdoor.to_csv(
                f"{output_dir}/inference_set-{pick_C}-irun-{iRun}_df_r56.csv"
            )

            df0_vanilla.to_csv(
                f"{args.output_dir_noBackdoor}/inference_set-{pick_C}-irun-{iRun}_df_avh.csv"
            )
            df1_vanilla.to_csv(
                f"{args.output_dir_noBackdoor}/inference_set-{pick_C}-irun-{iRun}_df_r56.csv"
            )
