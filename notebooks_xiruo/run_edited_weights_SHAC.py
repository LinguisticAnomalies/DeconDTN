import os
import argparse

### Temporary Argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="SHAC",
    help="Dataset for the experiment",
)
parser.add_argument("--weightsEdited", type=str, help="Path to edited weights")
parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
parser.add_argument(
    "-q", "--quantization", action="store_true", help="whether to use quantization"
)
parser.add_argument(
    "--nRuns", type=int, default=1, help="Number of experiments to run"
)
parser.add_argument(
    "--percent", type=int, default=5, help="1/X of total setting will be used"
)
parser.add_argument(
    "--sampleValidSettings", action="store_true", help="whether to sample valid settings"
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
parser.add_argument(
    "--cpuOps",
    action="store_true",
    help="Unload to CPU for tensors. This still stores state_dict() on GPU in the end",
)
parser.add_argument(
    "--mntdir",
    type=str,
    default="/bime-munin/",
    help="Number of testing samples",
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
sys.path.append("../config")

from utils import number_split, create_mix
from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC
from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from sampling_numbers import HateSpeech_DICT, SHAC_DICT

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
from accelerate.utils import load_and_quantize_model
from accelerate.utils import BnbQuantizationConfig
from accelerate import init_empty_weights



tmp = [x for x in args.weightsEdited.split("/") if "set-" in x]
name_pre = tmp[0].split(".pth")[0]  # of form like set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1-added.pth
model_size = int([x for x in name_pre.split("-") if "B" in x][0].replace("B", ""))  # 7, 13, 70
assert model_size in (7, 13, 70)



class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.model_id = f"{args.mntdir}/llama2_hf/llama-2-{model_size}b_hf/"
globalconfig.max_seq_length = 1024
globalconfig.device = args.device

##### Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(f"{args.mntdir}/llama2_hf/llama-2-7b_hf/", use_safetensors=False)

tokenizer.add_special_tokens({"pad_token": "<pad>"})

if args.cpuOps:
    load_device = "cpu"
    load_state_device = "cpu"
else:
    load_device = 'auto'
    load_state_device = globalconfig.device
##### Load Model and  Update using Edited Weights 
model = LlamaForSequenceClassification.from_pretrained(
    globalconfig.model_id,
    device_map=load_device,
    # load_in_8bit=args.quantization,
    # torch_dtype=torch.float16,
    use_safetensors=False
)

model.config.pad_token_id = tokenizer.pad_token_id

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

# this step cannot be ignored here...
model.load_state_dict(torch.load(args.weightsEdited, 
                                 map_location=load_state_device,
                                 # map_location=lambda storage, loc: storage,
                                ))

print("###  Finished Loading...")

if args.quantization:
    bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
    model = load_and_quantize_model(model, weights_location=args.weightsEdited, bnb_quantization_config=bnb_quantization_config, device_map = load_device)

print("###  Finished Quantization...")

######  Load Data
if args.dataset == "SHAC":
    ### SHAC
    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x["Drug"] else 0, axis=1)

    df_shac["dfSource"] = df_shac["location"]
    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)


elif args.dataset == "HateSpeech":
        
    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()



y_Categories = [0, 1]
n_yCats = len(y_Categories)


##### Split
# SHAC-Drug - Balanced Alpha
n_test = int([x for x in args.weightsEdited.split("/") if x.startswith("n")][0].strip("n"))
train_test_ratio = 4

if args.dataset == "SHAC":
    p_pos_train_z0_ls = SHAC_DICT["Run-0"]['p_pos_train_z0_ls']
    p_pos_train_z1_ls = SHAC_DICT["Run-0"]['p_pos_train_z1_ls']
    p_mix_z1_ls = SHAC_DICT["Run-0"]['p_mix_z1_ls']
elif args.dataset == "HateSpeech":
    p_pos_train_z0_ls = HateSpeech_DICT["Run-1"]['p_pos_train_z0_ls']
    p_pos_train_z1_ls = HateSpeech_DICT["Run-1"]['p_pos_train_z1_ls']
    p_mix_z1_ls = HateSpeech_DICT["Run-1"]['p_mix_z1_ls']

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


runs = args.nRuns

outdir = args.output_dir
name_general = f"OriginalWeightsEdited-{name_pre}-ntest_{n_test}-pct_1_{args.percent}"
log_f = f"../log/{name_general}.log"
_name_split = name_pre.split("-")  ## set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth
pick_C = int(_name_split[_name_split.index('set')+1])


if args.dataset == "HateSpeech":

    ### Hate Speech
    z_Categories = ["dynGen", "wsf"]  # the order here matters! Should match with df0, df1
    label = "label_binary"
    split_label = "label_binary"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "dfSource"
    df0 = df_dynGen
    df1 = df_wsf

    c = HateSpeech_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n1000_9870"

elif args.dataset == "SHAC":
    ### SHAC
    z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
    label = "label_binary"
    split_label = 'Drug'
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "location"
    df0 = df_shac_uw
    df1 = df_shac_mimic

    c = SHAC_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_2800"


dfs_used = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, 
                # seed=random.randint(0,1000),
                seed=222
                )

assert dfs_used is not None

df0 = df0[~df0[txt_col].isin(dfs_used['train'][txt_col])].reset_index(drop=True)
df0 = df0[~df0[txt_col].isin(dfs_used['test'][txt_col])].reset_index(drop=True)

    
df1 = df1[~df1[txt_col].isin(dfs_used['train'][txt_col])].reset_index(drop=True)
df1 = df1[~df1[txt_col].isin(dfs_used['test'][txt_col])].reset_index(drop=True)




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


record_valid_settings_n = []


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
    _n_setting = 0
    
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
        
        
        ##### NTOE: for shorter version!!!
        if args.sampleValidSettings:
            if round(c['mix_param_dict']['alpha_train'], 4) not in [1, 2, 0.5, 4, 0.25, 6, 0.1667]:
                continue

        _n_setting += 1
        if _n_setting % args.percent != 0:
            continue


        c["run"] = iRun
        record_valid_settings_n.append(c)

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


for k in record_valid_settings_n[0]["mix_param_dict"].keys():
    df_eval[k] = [_dict["mix_param_dict"][k] for _dict in record_valid_settings_n]

for k in record_valid_settings_n[0].keys():
    if k != "mix_param_dict":
        df_eval[k] = [_dict[k] for _dict in record_valid_settings_n]


outname = f"{outdir}/{name_general}.pkl"
with open(outname, "wb") as f:
    pickle.dump(df_eval, file=f)