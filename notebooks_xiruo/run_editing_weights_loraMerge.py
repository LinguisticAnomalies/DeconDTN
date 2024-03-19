import os
import argparse

### Temporary Argparse
parser = argparse.ArgumentParser()
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
    "-q", "--quantization", action="store_true", help="whether to use quantization"
)
parser.add_argument(
    "--gamma", type=int, default=1, help="scaling parameter for delta weight matrices"
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="On which GPU to run",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import peft
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
    PeftModel,
)
from peft.utils import (
    _freeze_adapter,
    _get_submodules,
)
from peft.tuners.lora import LoraLayer
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
    LlamaTokenizer,
    LlamaForSequenceClassification,
    TrainerCallback,
    default_data_collator,
)
from torch.linalg import vector_norm
from torch.linalg import matrix_norm
import random


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
model_size = int(name_pre.split("-")[-3].replace("B", ""))  # 7, 13, 70
assert model_size in (7, 13, 70)

model_id = f"/bime-munin/llama2_hf/llama-2-{model_size}b_hf/"
weights_delta_file = f"{args.weightsEditedDir}/{os.path.basename(target_model_id)}-delta.pth"        
weights_edited_file = f"{args.weightsEditedDir}/{os.path.basename(target_model_id)}-gamma_1-added.pth"


##### Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(f"/bime-munin/llama2_hf/llama-2-7b_hf/")

tokenizer.add_special_tokens({"pad_token": "<pad>"})

# ##### Load Target Adapter, Merge and Unload 

# base_model = LlamaForSequenceClassification.from_pretrained(model_id, device_map='auto', load_in_8bit=args.quantization, torch_dtype=torch.float16)

# base_model.config.pad_token_id = tokenizer.pad_token_id
# base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

# model = PeftModel.from_pretrained(base_model, source_model_id, adapter_name='target')
# merged_Target_model = model.merge_and_unload(progressbar=True)

# state_dict_T = merged_Target_model.state_dict()


# ##### Load Source Adapter, Merge and Unload 
# base_model = LlamaForSequenceClassification.from_pretrained(model_id, device_map='auto', load_in_8bit=args.quantization, torch_dtype=torch.float16)

# base_model.config.pad_token_id = tokenizer.pad_token_id
# base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

# model = PeftModel.from_pretrained(base_model, target_model_id, adapter_name='source')
# merged_Source_model = model.merge_and_unload(progressbar=True)

# state_dict_S = merged_Source_model.state_dict()


# ##### Calculate Weight Delta & Save

# for k in state_dict_T.keys():
#     if k.endswith(".weight"):
#         state_dict_T[k] = state_dict_T[k] - state_dict_S[k]

# # args.weightsEditedDir = "/bime-munin/xiruod/llama2_SHAC/n500/Weights/"
# torch.save(state_dict_T, weights_delta_file)

# print("Successfully Saving Delta Weights!!!")


# del state_dict_T, state_dict_S, base_model, model, merged_Target_model, merged_Source_model
# gc.collect()
# torch.cuda.empty_cache() 


##### Load Delta and Merge with Original Pretrained Model

state_dict_delta = torch.load(weights_delta_file)


base_model = LlamaForSequenceClassification.from_pretrained(model_id, device_map='auto', load_in_8bit=args.quantization)

base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

state_dict_o = base_model.state_dict()


for k in state_dict_o.keys():
    if k.endswith(".weight"):
        state_dict_o[k] = state_dict_o[k] + args.gamma * state_dict_delta[k]


torch.save(state_dict_o, weights_edited_file)


print("Successfully Edited Weights!!!")