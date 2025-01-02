import os
import argparse

parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument(
    "--dataset",
    type=str,
    default="SHAC",
    help="Dataset for the experiment",
)
parser.add_argument("-c", "--CombinationIdx", type=int, help="Set idx of c to use")
parser.add_argument("--model_name", default="", help="Model to use. Default RoBERTa")
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
    help="cuda device",
)
parser.add_argument(
    "--nTest",
    type=int,
    default=200,
    help="Number of testing samples",
)
parser.add_argument("--batchSize", type=int, default=8, help="Batch size")
parser.add_argument(
    "--mntdir",
    type=str,
    default="/bime-munin/",
    help="Number of testing samples",
)
parser.add_argument("--grad_reverse", action="store_true")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import sys
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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainerCallback,
    default_data_collator,
)

sys.path.append("../src")
sys.path.append("../config")

from utils import number_split, create_mix, confoundSplitDF
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT

from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_SHAC import load_process_SHAC
from process_CD import load_cd
from AdversarialModel import GradientReverseModel


class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.model_name = args.model_name
globalconfig.max_seq_length = 512
globalconfig.num_train_epochs = 3
globalconfig.runs = 1
globalconfig.lr = 1e-4
globalconfig.warmup_ratio = 0.1
globalconfig.profiler = False
globalconfig.device = args.device
globalconfig.per_device_train_batch_size = args.batchSize
globalconfig.per_device_eval_batch_size = args.batchSize


##### Split Settings
n_test = args.nTest
train_test_ratio = 4

pick_C = args.CombinationIdx

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

globalconfig.output_dir = f"{args.mntdir}/xiruod/GradientReverse_{args.model_name}_{args.dataset}/n{args.nTest}/"

if args.dataset == "SHAC":
    label = "Drug"

    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x[label] else 0, axis=1)
    df_shac["dfSource"] = df_shac[domain_col]

elif args.dataset == "HateSpeech":
    label = "label"
    ## Hate Speech data already have "label_binary" and dfSource
    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()
elif args.dataset == "CD":
    label = "label"

    df_all = load_cd()
    df_avh = df_all["avh"]
    df_r56 = df_all["r56"]

label2id = {z: idx for idx, z in zip(range(len(y_Categories)), y_Categories)}
id2label = {idx: z for idx, z in zip(range(len(y_Categories)), y_Categories)}


if args.dataset == "SHAC":
    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

    df0 = df_shac_uw
    df1 = df_shac_mimic
    df_split_label = "Drug"

    c = SHAC_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_2800"


elif args.dataset == "HateSpeech":
    df0 = df_dynGen
    df1 = df_wsf
    df_split_label = "label_binary"

    c = HateSpeech_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n1000_9870"

elif args.dataset == "CD":
    df0 = df_avh
    df1 = df_r56
    df_split_label = "label_binary"

    c = CD_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_566"


# run for check valid settings

import warnings

warnings.simplefilter("ignore")

df0["domain_index"] = 0
df1["domain_index"] = 1

##### Experiment - ONLY One Setting

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


X_train = dfs["train"]["text"]
y_train = dfs["train"][["label"]]
y_domain_train = dfs["train"][["domain_index"]]

## Initialize model
torch.manual_seed(222)
torch.cuda.manual_seed(222)
torch.cuda.manual_seed_all(222)


model_config = {}
# model_config['model_type'] = model_type
model_config["pretrained"] = globalconfig.model_name
model_config["max_length"] = globalconfig.max_seq_length
model_config["num_labels"] = len(y_Categories)
model_config["num_domain_labels"] = len(z_category)
model_config["hidden_dropout_prob"] = 0.1
model_config["num_epochs"] = globalconfig.num_train_epochs
model_config["num_warmup_steps"] = 0
model_config["batch_size"] = args.batchSize
model_config["lr"] = globalconfig.lr
model_config["balance_weights"] = False
model_config["grad_norm"] = 1.0
model_config["grad_reverse"] = args.grad_reverse

model = GradientReverseModel(**model_config)

model.load_pretrained()

# train & predict
model.trainModel(
    X=X_train,
    y=y_train,
    y_domain=y_domain_train,
    device=args.device,
)


output_dir = globalconfig.output_dir
os.makedirs(output_dir, exist_ok=True)

outfile = f"{output_dir}/set-{args.CombinationIdx}-epoch{globalconfig.num_train_epochs}.pth"

torch.save(model, outfile)
