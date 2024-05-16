import os
import argparse

### Temporary Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="", help="Model to use. Default RoBERTa")
parser.add_argument(
    "--target_model_id", type=str, help="Directory to the Target adapter"
)
parser.add_argument(
    "--source_model_id",
    type=str,
    default=None,
    help="Directory to the Source adapter. If None, then set to target_model_id with prefix 'Source-'.",
)
parser.add_argument(
    "--weightsEditedDir", type=str, default=None, help="Dir to edited weights"
)
parser.add_argument(
    "--lambda1",
    type=float,
    default=1,
    help="scaling parameter for delta weight matrices",
)
parser.add_argument(
    "--lambda2",
    type=float,
    default=1,
    help="scaling parameter for delta weight matrices",
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
parser.add_argument(
    "--mntdir",
    type=str,
    default="/bime-munin/",
    help="Number of testing samples",
)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


from dataclasses import asdict, replace
from functools import reduce
import operator
import sys
import gc

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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainerCallback,
    default_data_collator,
)
from torch.linalg import vector_norm
from torch.linalg import matrix_norm
import random
from copy import deepcopy

target_model_id = (
    args.target_model_id
)  # "/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8"

if args.source_model_id is not None:
    source_model_id = (
        args.source_model_id
    )  # "/bime-munin/xiruod/llama2_SHAC/n500/Source-set-1355-quantization-epoch3-llama-2-7B-loraR-8"
else:
    nm_split = target_model_id.strip().split("/")
    nm_mod = [x if "set-" not in x else "Source-" + x for x in nm_split]
    source_model_id = "/".join(nm_mod)

tmp = [x for x in target_model_id.split("/") if "set-" in x]
name_pre = tmp[0]  # of form like set-1355-quantization-epoch3-llama-2-7B-loraR-8

weights_edited_file = f"{args.weightsEditedDir}/{os.path.basename(target_model_id)}-lambda1_{args.lambda1:.1f}-lambda2_{args.lambda2:.1f}-added.pth"

os.makedirs(args.weightsEditedDir, exist_ok=True)


def amplifyWeights(model_in, magnitude=1.0):
    ret = {}
    for wname, W in model_in.named_parameters():
        if ("query" in wname) or ("value" in wname) or ("classifier" in wname):
            W.data = (W.data - state_dict_oT[wname]) * magnitude

            ret[wname] = W.data
    return ret


##### Load Target
torch.manual_seed(222)
torch.cuda.manual_seed(222)
torch.cuda.manual_seed_all(222)
base_model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    use_safetensors=False,
)

state_dict_oT = deepcopy(base_model.state_dict())


target_model = AutoModelForSequenceClassification.from_pretrained(
    target_model_id,
    use_safetensors=False,
)

vector_target = amplifyWeights(target_model, magnitude=args.lambda1)


source_model = AutoModelForSequenceClassification.from_pretrained(
    source_model_id,
    use_safetensors=False,
)

vector_source = amplifyWeights(source_model, magnitude=args.lambda2)

assert set(vector_target.keys()) == set(vector_source.keys())


for k in state_dict_oT.keys():
    if k in set(vector_target.keys()):
        state_dict_oT[k] = vector_target[k] - vector_source[k] + state_dict_oT[k]


torch.save(state_dict_oT, weights_edited_file)

print("Successfully Edited Weights!!!")
