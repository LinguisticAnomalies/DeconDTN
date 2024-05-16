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
    "--lambda1",
    type=float,
    default=1,
    help="scaling parameter for delta weight matrices",
)
parser.add_argument(
    "--DeltaFinished", action="store_true", help="if delta weights have been calculated"
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
from safetensors import safe_open
from torch.linalg import vector_norm
from torch.linalg import matrix_norm
from numpy.linalg import lstsq
from scipy.linalg import orth
import random
from copy import deepcopy


def find_orth(in_vectors):
    rand_vec = np.random.rand(in_vectors.shape[0], 1)
    A = np.hstack((in_vectors, rand_vec))
    b = np.zeros(in_vectors.shape[1] + 1)
    b[-1] = 1
    return lstsq(A.T, b)[0]


def projVector(a, projv):

    ret = (
        a.flatten().numpy()
        - np.dot(a.flatten().numpy(), projv) / (np.linalg.norm(projv) ** 2) * projv
    )
    return torch.tensor(ret.reshape(a.shape)).type(torch.float32)


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

    reverse_nm_mod = [x if "set-" not in x else "Reverse-Source-" + x for x in nm_split]
    source_model_id_2 = "/".join(reverse_nm_mod)


tmp = [x for x in target_model_id.split("/") if "set-" in x]
name_pre = tmp[0]  # of form like set-1355-quantization-epoch3-llama-2-7B-loraR-8
model_size = int(name_pre.split("-")[-3].replace("B", ""))  # 7, 13, 70
assert model_size in (7, 13, 70)

model_id = f"/{args.mntdir}/llama2_hf/llama-2-{model_size}b_hf/"
weights_edited_file = f"{args.weightsEditedDir}/{os.path.basename(target_model_id)}-lambda1_{args.lambda1}-added-Norm.pth"

os.makedirs(f"{args.weightsEditedDir}", exist_ok=True)


##### Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(
    f"/{args.mntdir}/llama2_hf/llama-2-7b_hf/", use_safetensors=False
)

tokenizer.add_special_tokens({"pad_token": "<pad>"})


def amplifyLoraWeights(model_in, adapter_name, magnitude=1.0):
    key_list = [
        key for key, _ in model_in.model.named_modules() if model_in.prefix not in key
    ]
    for key in key_list:
        _, target, _ = _get_submodules(model_in.model, key)
        if isinstance(target, LoraLayer):
            if adapter_name in target.lora_A:
                target_lora_A = target.lora_A[adapter_name].weight
                target_lora_B = target.lora_B[adapter_name].weight
            elif adapter_name in target.lora_embedding_A:
                target_lora_A = target.lora_embedding_A[adapter_name]
                target_lora_B = target.lora_embedding_B[adapter_name]
            else:
                continue

            # Weights should be only amplified once, e.g., magnitude ^ 1, instead of magnitude ^ 2
            target_lora_A.data = target_lora_A.data * magnitude
            target_lora_B.data = target_lora_B.data

    return model_in


def amplifyLoraWeightsTowardsTarget(
    model_in,
    adapter_name,
    target_tensors,
    source_tensors,
    reverse_source_tensors,
):

    key_list = [
        key for key, _ in model_in.model.named_modules() if model_in.prefix not in key
    ]
    for key in key_list:
        _, target, _ = _get_submodules(model_in.model, key)
        if isinstance(target, LoraLayer):
            if adapter_name in target.lora_A:
                target_lora_A = target.lora_A[adapter_name].weight
                target_lora_B = target.lora_B[adapter_name].weight
            elif adapter_name in target.lora_embedding_A:
                target_lora_A = target.lora_embedding_A[adapter_name]
                target_lora_B = target.lora_embedding_B[adapter_name]
            else:
                continue

            # Weights should be only amplified once, e.g., magnitude ^ 1, instead of magnitude ^ 2
            k = "base_model.model." + key

            A_t = target_tensors[f"{k}.lora_A.weight"]
            B_t = target_tensors[f"{k}.lora_B.weight"]

            A_s = source_tensors[f"{k}.lora_A.weight"]
            B_s = source_tensors[f"{k}.lora_B.weight"]

            A_rs = reverse_source_tensors[f"{k}.lora_A.weight"]
            B_rs = reverse_source_tensors[f"{k}.lora_B.weight"]

            norm_t = torch.norm(torch.matmul(B_t, A_t), p="fro").item()
            norm_s = torch.norm(torch.matmul(B_s, A_s), p="fro").item()
            norm_rs = torch.norm(torch.matmul(B_rs, A_rs), p="fro").item()

            magnitude = (args.lambda1 - 1) / (
                norm_s / norm_t + norm_rs / norm_t
            )  # a/(p1+p2)

            # Weights should be only amplified once, e.g., magnitude ^ 1, instead of magnitude ^ 2
            target_lora_A.data = target_lora_A.data * magnitude
            target_lora_B.data = target_lora_B.data

    return model_in


##### Readin Lora Adapters for all 3 models
target_tensors = {}
source_tensors = {}
reverse_source_tensors = {}


with safe_open(
    f"{target_model_id}/adapter_model.safetensors", framework="pt", device="cpu"
) as f:
    for key in f.keys():
        target_tensors[key] = f.get_tensor(key)

with safe_open(
    f"{source_model_id}/adapter_model.safetensors", framework="pt", device="cpu"
) as f:
    for key in f.keys():
        source_tensors[key] = f.get_tensor(key)

with safe_open(
    f"{source_model_id_2}/adapter_model.safetensors", framework="pt", device="cpu"
) as f:
    for key in f.keys():
        reverse_source_tensors[key] = f.get_tensor(key)


##### Load Target Adapter, Merge and Unload
if not args.DeltaFinished:
    if args.cpuOps:
        load_device = "cpu"
    else:
        load_device = "auto"

    ##### Load Target Adapter, Merge and Unload
    torch.manual_seed(222)
    torch.cuda.manual_seed(222)
    torch.cuda.manual_seed_all(222)

    base_model = LlamaForSequenceClassification.from_pretrained(
        model_id,
        device_map=load_device,
        use_safetensors=False,
    )  # load_in_8bit=args.quantization, torch_dtype=torch.bfloat16 if args.quantization else torch.float32)

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
    state_dict_oT = deepcopy(base_model.state_dict())

    model = PeftModel.from_pretrained(
        base_model, target_model_id, adapter_name="target", use_safetensors=False
    )
    score_weight_vector = (
        base_model.state_dict()["score.modules_to_save.target.weight"]
        - base_model.state_dict()["score.original_module.weight"]
    )

    lora_model_keys = model.state_dict().keys()

    model = amplifyLoraWeights(
        model_in=model, adapter_name="target", magnitude=args.lambda1
    )

    merged_Target_model = model.merge_and_unload(progressbar=True)

    state_dict_T = merged_Target_model.state_dict()
    state_dict_T["score.weight"] = score_weight_vector * args.lambda1

    ##### Load Source Adapter, Merge and Unload
    torch.manual_seed(222)
    torch.cuda.manual_seed(222)
    torch.cuda.manual_seed_all(222)

    base_model = LlamaForSequenceClassification.from_pretrained(
        model_id,
        device_map=load_device,
        use_safetensors=False,
    )  # load_in_8bit=args.quantization, torch_dtype=torch.bfloat16 if args.quantization else torch.float32)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

    model = PeftModel.from_pretrained(
        base_model, source_model_id, adapter_name="source", use_safetensors=False
    )
    score_weight_vector = (
        base_model.state_dict()["score.modules_to_save.source.weight"]
        - base_model.state_dict()["score.original_module.weight"]
    )

    model = amplifyLoraWeightsTowardsTarget(
        model_in=model,
        adapter_name="source",
        target_tensors=target_tensors,
        source_tensors=source_tensors,
        reverse_source_tensors=reverse_source_tensors,
    )

    lambda2_cls = (args.lambda1 - 1) / 2

    merged_Source_model = model.merge_and_unload(progressbar=True)

    state_dict_S = merged_Source_model.state_dict()
    state_dict_S["score.weight"] = score_weight_vector * lambda2_cls

    del base_model, model, merged_Target_model, merged_Source_model

    ##### Load Source-2 Adapter, Merge and Unload
    torch.manual_seed(222)
    torch.cuda.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    base_model = LlamaForSequenceClassification.from_pretrained(
        model_id,
        device_map=load_device,
        use_safetensors=False,
    )  # load_in_8bit=args.quantization, torch_dtype=torch.bfloat16 if args.quantization else torch.float32)

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

    model = PeftModel.from_pretrained(
        base_model,
        source_model_id_2,
        adapter_name="source_reverse",
        use_safetensors=False,
    )
    score_weight_vector = (
        base_model.state_dict()["score.modules_to_save.source_reverse.weight"]
        - base_model.state_dict()["score.original_module.weight"]
    )

    model = amplifyLoraWeightsTowardsTarget(
        model_in=model,
        adapter_name="source_reverse",
        target_tensors=target_tensors,
        source_tensors=source_tensors,
        reverse_source_tensors=reverse_source_tensors,
    )

    lambda3_cls = (args.lambda1 - 1) / 2

    merged_Source_Reverse_model = model.merge_and_unload(progressbar=True)

    state_dict_S_reverse = merged_Source_Reverse_model.state_dict()
    state_dict_S_reverse["score.weight"] = score_weight_vector * lambda3_cls

    del base_model, model, merged_Source_Reverse_model

    ##### Calculate Weight Delta & Save
    lora_layers = set(
        [
            x.split("base_model.model.")[1].replace("base_layer.", "")
            for x in lora_model_keys
            if "lora" in x
        ]
    )
    lora_layers_MapOriginalNames = set(
        [x.split(".lora")[0] + ".weight" for x in lora_layers]
    )

    for k in state_dict_oT.keys():
        if k in list(lora_layers_MapOriginalNames) + ["score.weight"]:
            state_dict_oT[k] = (
                state_dict_T[k]
                - state_dict_S[k]
                - state_dict_S_reverse[k]
                + state_dict_oT[k] * 2
            )
    if args.cpuOps:
        for k in state_dict_oT.keys():
            if k.endswith(".weight"):
                state_dict_oT[k] = state_dict_oT[k].to("cuda:0")

    torch.save(state_dict_oT, weights_edited_file)

    print("Successfully Edited Weights!!!")
