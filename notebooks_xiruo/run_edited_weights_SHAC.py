import os
import argparse

### Temporary Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--weightsEdited", type=str, help="Path to edited weights")
parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
parser.add_argument(
    "-q", "--quantization", action="store_true", help="whether to use quantization"
)
parser.add_argument(
    "--percent", type=int, default=5, help="1/X of total setting will be used"
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
    "--device",
    type=str,
    default="cuda:0",
    help="Specify cuda GPU",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
)
from peft import PeftModel

import sys

sys.path.append("../src")

from utils import number_split, create_mix
from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC

import itertools
from tqdm.auto import tqdm
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import datasets
from contextlib import nullcontext
import torch
from torch import nn
from transformers import (
    Trainer,
    TrainingArguments,
    LlamaTokenizer,
    LlamaForSequenceClassification,
    TrainerCallback,
    default_data_collator,
)
import random
from sklearn import metrics
from scipy.special import softmax
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


tmp = [x for x in args.weightsEdited.split("/") if "set-" in x]
name_pre = tmp[0].split(".")[0]  # of form like set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1-added.pth
model_size = int([x for x in name_pre.split("-") if "B" in x][0].replace("B", ""))  # 7, 13, 70
assert model_size in (7, 13, 70)


class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.model_id = f"/bime-munin/llama2_hf/llama-2-{model_size}b_hf/"
globalconfig.max_seq_length = 1024
globalconfig.device = args.device

##### Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(f"/bime-munin/llama2_hf/llama-2-7b_hf/")

tokenizer.add_special_tokens({"pad_token": "<pad>"})

##### Load Model and  Update using Edited Weights 
model = LlamaForSequenceClassification.from_pretrained(
    globalconfig.model_id,
    device_map="auto",
    load_in_8bit=args.quantization,
    # torch_dtype=torch.float16,
)

model.config.pad_token_id = tokenizer.pad_token_id

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)


model.load_state_dict(torch.load(args.weightsEdited, map_location="cuda:0"))


######  Load Data
### SHAC
df_shac = load_process_SHAC(replaceNA="all")
df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x["Drug"] else 0, axis=1)

df_shac["dfSource"] = df_shac["location"]
df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

y_Categories = [0, 1]
n_yCats = len(y_Categories)


##### Split
# SHAC-Drug - Balanced Alpha
n_test = 500
train_test_ratio = 4


p_pos_train_z0_ls = np.arange(
    0, 1, 0.1
)  # probability of training set examples drawn from site/domain z0 being positive
p_pos_train_z1_ls = np.arange(
    0, 1, 0.1
)  # probability of test set examples drawn from site/domain z1 being positive


p_mix_z1_ls = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]  # = np.arange(0.1, 0.9, 0.05)

# alpha_test_ls = np.arange(0, 10, 0.05)

numvals = 1023
base = 1.1
alpha_test_ls = np.power(base, np.arange(numvals)) / np.power(base, numvals // 2)


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


##### Run Experiments
import warnings

warnings.simplefilter("ignore")


runs = 1


# ### Hate Speech
# z_Categories = ["dynGen", "wsf"]  # the order here matters! Should match with df0, df1
# label = "label_binary"
# n_zCats = len(z_Categories)
# txt_col = "text"
# domain_col = "dfSource"
# df0 = df_dynGen
# df1 = df_wsf
# outdir = f"../output/DistilBERT_HateSpeechBalanceAlpha_RandomPermute_ShorterVersion_1_5"
# log_f = "../log/DistilBERT_HateSpeechBalanceAlpha_RandomPermute_ShorterVersion_1_5.log"


### SHAC
z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
label = "label_binary"
n_zCats = len(z_Categories)
txt_col = "text"
domain_col = "location"
df0 = df_shac_uw
df1 = df_shac_mimic
outdir = args.output_dir
name_general = f"{name_pre}_ntest_{n_test}_setting_1_{args.percent}"
log_f = f"../log/{name_general}.log"
# log_f = f"../log/test.log"


# NTOE: for shorter version!!!
valid_full_settings = [
    valid_full_settings[x]
    for x in list(range(len(valid_full_settings)))
    if x % args.percent == 0
]

# if args.save_model:
#     valid_full_settings = [
#         setting
#         for setting in valid_full_settings
#         if (
#             (round(setting["mix_param_dict"]["alpha_train"], 1) in (0.3, 3.0))
#             and (
#                 (0.20 <= round(setting["mix_param_dict"]["alpha_test"], 2) <= 0.30)
#                 or (1.30 <= round(setting["mix_param_dict"]["alpha_test"], 2) <= 1.40)
#                 or (3.7 <= round(setting["mix_param_dict"]["alpha_test"], 2) <= 3.8)
#             )
#             and (round(setting["mix_param_dict"]["C_y"], 2) in (0.24, 0.36, 0.48))
#         )
#     ]
#     outdir = f"../output/DistilBERT_SHACBalanceAlpha_RandomPermute_ShorterVersion_1_5_SaveModel"
#     log_f = ("../log/DistilBERT_SHACBalanceAlpha_RandomPermute_ShorterVersion_1_5_SaveModel.log")


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


# setting for logistic regression
# penalty = "l1"
# solver = "liblinear"
# penalty = "l2"
# solver = "lbfgs"


random.seed(123)
auprc_weightsEdited = []
auprc_weightsEdited_df0 = []
auprc_weightsEdited_df1 = []


valid_n_full_settings = []


precision_weightsEdited = []
recall_weightsEdited = []
f1_weightsEdited = []
precision_weightsEdited_df0 = []
recall_weightsEdited_df0 = []
f1_confounder_df0 = []
precision_weightsEdited_df1 = []
recall_weightsEdited_df1 = []
f1_weightsEdited_df1 = []

# precision_vanilla = []
# recall_vanilla = []
# f1_vanilla = []
# precision_vanilla_df0 = []
# recall_vanilla_df0 = []
# f1_vanilla_df0 = []
# precision_vanilla_df1 = []
# recall_vanilla_df1 = []
# f1_vanilla_df1 = []


##### Dataset Loader and Tokenizer
def preprocess_function(examples):
    # tokenize
    ret = tokenizer(
        examples[txt_col],
        return_tensors="pt",
        max_length=globalconfig.max_seq_length,
        padding="max_length",
        truncation=True,
    ).to(globalconfig.device)

    return ret


def datasets_loader(df):
    # from pandas df to Dataset & tokenize
    ret_datasets = datasets.Dataset.from_pandas(
        df[[txt_col, "dfSource", "label_binary"]]
        .rename(columns={"label_binary": "label"})
        .reset_index(drop=True)
    )
    ret_tokenized = ret_datasets.map(preprocess_function, batched=True)

    return ret_tokenized


for iRun in range(runs):
    _rand = random.randint(0, 2**32 - 1)
    print(_rand)

    print(iRun)
    for c in tqdm(valid_full_settings, file=open(log_f, "w")):
        # for c in test_settings:

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

        y_train = dfs["train"][label]
        y_test = dfs["test"][label]

        n_test = len(y_test)
        df_test = dfs["test"].copy(deep=True)

        # tokenized_train = datasets_loader(dfs["train"])
        df_in = tokenizer(
            list(dfs["test"]["text"]),
            return_tensors="pt",
            max_length=globalconfig.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        y_ls = []
        lst = list(range(len(df_in["input_ids"])))
        n = args.batch_size
        idx_ls = [lst[i : i + n] for i in range(len(lst)) if i % n == 0]

        model.eval()
        with torch.no_grad():
            for idx in idx_ls:
                ret_output = model.forward(
                    input_ids=df_in["input_ids"][idx].to(globalconfig.device),
                    attention_mask=df_in["attention_mask"][idx].to(globalconfig.device),
                )
                y_probs_ = softmax(ret_output["logits"].cpu(), axis=1)
                y_ls.append(y_probs_)
            y_probs_auprc_weightsEdited = np.concatenate(y_ls)

        # ret_eval = trainer.evaluate()

        # Predict

        ## SR0, SR1, SR2, ...

        # ret_pred = trainer.predict(test_dataset=df_test["tokenized"])
        # y_probs_confound = softmax(ret_pred.predictions, axis=1)

        # y_probs_ls = []

        # for i in range(n_zCats):
        #     _y_probs = trainer.predict(test_dataset=df_test[f"tokenized_sr{i}"])
        #     _y_probs = softmax(_y_probs.predictions, axis=1)
        #     y_probs_ls.append(_y_probs)

        # # calculate P(Z): NOTE: this may not be useful, because it is pre-defined!!!!
        # p_z = []

        # for i in z_Categories:
        #     p_z.append(sum(dfs["train"][domain_col] == i) / len(dfs["train"]))

        # calculate P(Y|X): sum(P(y|x,z) * P(z))
        # y_probs_confound = np.empty((n_test, n_yCats))
        # y_probs_confound.fill(0)

        # for i in range(n_zCats):
        #     y_probs_confound += y_probs_ls[i] * p_z[i]

        # save metrics
        ret = c

        ret_code = 1

        auprc_weightsEdited.append(
            metrics.average_precision_score(y_true=y_test, y_score=y_probs_auprc_weightsEdited[:, 1])
        )
        auprc_weightsEdited_df0.append(
            metrics.average_precision_score(
                y_true=y_test[df_test[domain_col] == z_Categories[0]],
                y_score=y_probs_auprc_weightsEdited[df_test[domain_col] == z_Categories[0], 1],
            )
        )
        auprc_weightsEdited_df1.append(
            metrics.average_precision_score(
                y_true=y_test[df_test[domain_col] == z_Categories[1]],
                y_score=y_probs_auprc_weightsEdited[df_test[domain_col] == z_Categories[1], 1],
            )
        )
        t = precision_recall_fscore_support(
            y_true=y_test,
            y_pred=y_probs_auprc_weightsEdited[:, 1] > 0.5,
            average="binary",
            pos_label=1,
        )
        t_df0 = precision_recall_fscore_support(
            y_true=y_test[df_test[domain_col] == z_Categories[0]],
            y_pred=y_probs_auprc_weightsEdited[df_test[domain_col] == z_Categories[0], 1] > 0.5,
            average="binary",
            pos_label=1,
        )
        t_df1 = precision_recall_fscore_support(
            y_true=y_test[df_test[domain_col] == z_Categories[1]],
            y_pred=y_probs_auprc_weightsEdited[df_test[domain_col] == z_Categories[1], 1] > 0.5,
            average="binary",
            pos_label=1,
        )
        precision_weightsEdited.append(t[0])
        recall_weightsEdited.append(t[1])
        f1_weightsEdited.append(t[2])
        precision_weightsEdited_df0.append(t_df0[0])
        recall_weightsEdited_df0.append(t_df0[1])
        f1_confounder_df0.append(t_df0[2])
        precision_weightsEdited_df1.append(t_df1[0])
        recall_weightsEdited_df1.append(t_df1[1])
        f1_weightsEdited_df1.append(t_df1[2])


############  Put Results in DataFrame, with extra information (a little redundant)

# organize results in DataFrame
df_eval = pd.DataFrame(
    {
        "auprc_weightsEdited": auprc_weightsEdited,
        "auprc_weightsEdited_df0": auprc_weightsEdited_df0,
        "auprc_weightsEdited_df1": auprc_weightsEdited_df1,
        "precision_weightsEdited": precision_weightsEdited,
        "recall_weightsEdited": recall_weightsEdited,
        "f1_weightsEdited": f1_weightsEdited,
        "precision_weightsEdited_df0": precision_weightsEdited_df0,
        "recall_weightsEdited_df0": recall_weightsEdited_df0,
        "f1_confounder_df0": f1_confounder_df0,
        "precision_weightsEdited_df1": precision_weightsEdited_df1,
        "recall_weightsEdited_df1": recall_weightsEdited_df1,
        "f1_weightsEdited_df1": f1_weightsEdited_df1,
        # "auprc_logistic_vanilla_df0": auprc_logistic_vanilla_df0,
        # "auprc_logistic_vanilla_df1": auprc_logistic_vanilla_df1,
        # "precision_vanilla":precision_vanilla,
        # "recall_vanilla":recall_vanilla,
        # "f1_vanilla":f1_vanilla,
        # "precision_vanilla_df0":precision_vanilla_df0,
        # "recall_vanilla_df0":recall_vanilla_df0,
        # "f1_vanilla_df0":f1_vanilla_df0,
        # "precision_vanilla_df1":precision_vanilla_df1,
        # "recall_vanilla_df1":recall_vanilla_df1,
        # "f1_vanilla_df1":f1_vanilla_df1,
    }
)


for k in valid_n_full_settings[0]["mix_param_dict"].keys():
    df_eval[k] = [_dict["mix_param_dict"][k] for _dict in valid_n_full_settings]

for k in valid_n_full_settings[0].keys():
    if k != "mix_param_dict":
        df_eval[k] = [_dict[k] for _dict in valid_n_full_settings]


outname = f"{outdir}/{name_general}.pkl"
with open(outname, "wb") as f:
    pickle.dump(df_eval, file=f)
