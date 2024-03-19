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
    "--adapterDir", type=str, default=None, help="Directory to the adapter."
)
parser.add_argument(
    "-q", "--quantization", action="store_true", help="whether to use quantization"
)
parser.add_argument(
    "--targetNorm", action="store_true", help="normalize to Target vectors"
)
parser.add_argument(
    "--targetFroNorm", action="store_true", help="normalize to Target vectors, Frobenius Norm"
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

##### Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(f"/bime-munin/llama2_hf/llama-2-7b_hf/")

tokenizer.add_special_tokens({"pad_token": "<pad>"})

##### Load Model and  Adapter
model = LlamaForSequenceClassification.from_pretrained(
    model_id,
    device_map="auto",
    load_in_8bit=args.quantization,
    torch_dtype=torch.float16,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

# Load Adapters
model = PeftModel.from_pretrained(model, target_model_id, adapter_name="target")

model.load_adapter(source_model_id, adapter_name="source")


######################  Edit Adapters
# Reference: function add_weighted_adapter from peft.tuners.lora.model
# https://github.com/huggingface/peft/blob/6008f272a565f56c146c5d9fd78d00cb24392d7b/src/peft/tuners/lora/model.py#L421

adapter_name = "delta"
adapters = ["target", "source"]

adapters_ranks = [model.peft_config[adapter].r for adapter in adapters]
new_rank = adapters_ranks[0]

target_module_types = [
    type(model.peft_config[adapter].target_modules) for adapter in adapters
]
if target_module_types[0] == str:
    new_target_modules = "|".join(
        f"({model.peft_config[adapter].target_modules})" for adapter in adapters
    )
elif target_module_types[0] == set:
    new_target_modules = reduce(
        operator.or_,
        (model.peft_config[adapter].target_modules for adapter in adapters),
    )


model.peft_config[adapter_name] = replace(
    model.peft_config[adapters[0]],
    r=new_rank,
    lora_alpha=new_rank,
    target_modules=new_target_modules,
)
model.inject_adapter(model.model, adapter_name)

# Do we really need that?
_freeze_adapter(model.model, adapter_name)

key_list = [key for key, _ in model.model.named_modules() if model.prefix not in key]
for key in key_list:
    _, target, _ = _get_submodules(model.model, key)
    if isinstance(target, LoraLayer):
        if adapter_name in target.lora_A:
            target_lora_A = target.lora_A[adapter_name].weight
            target_lora_B = target.lora_B[adapter_name].weight
        elif adapter_name in target.lora_embedding_A:
            target_lora_A = target.lora_embedding_A[adapter_name]
            target_lora_B = target.lora_embedding_B[adapter_name]
        else:
            continue

        target_lora_A.data = target_lora_A.data * 0.0
        target_lora_B.data = target_lora_B.data * 0.0

        target_lora_A.data = (
            target.lora_A["target"].weight - target.lora_A["source"].weight
        )
        target_lora_B.data = (
            target.lora_B["target"].weight - target.lora_B["source"].weight
        )
        
        if args.targetNorm:
            vt_oA = vector_norm(target_lora_A.data, dim=0)
            vt_tA = vector_norm(target.lora_A['target'].weight, dim=0)
            target_lora_A.data = target_lora_A.data / vt_oA * vt_tA

            vt_oB = vector_norm(target_lora_B.data, dim=1)
            vt_tB = vector_norm(target.lora_B['target'].weight, dim=1)
            target_lora_B.data = (target_lora_B.data.T / vt_oB * vt_tB).T
        
        elif args.targetFroNorm:
            target_lora_A.data = target_lora_A.data / matrix_norm(target_lora_A.data) * matrix_norm(target.lora_A['target'].weight)
            target_lora_B.data = target_lora_B.data / matrix_norm(target_lora_B.data) * matrix_norm(target.lora_B['target'].weight)


## Cleanup
model.delete_adapter("target")
model.delete_adapter("source")

## Save (only save 'delta')
# args.adapterDir = "../output/tmpData/set-1355-quantization-epoch3-llama-2-7B-loraR-8/delta/"
model.save_pretrained(args.adapterDir)

print("Successfully Edited Lora Weights!!!")