#!/usr/bin/env python
# coding: utf-8

import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

import seaborn as sns

from scipy.stats import wasserstein_distance
from itertools import combinations

import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys

sys.path.append("../src")
sys.path.append("../config")

from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC
from process_CD import load_cd
from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT

from tqdm.auto import tqdm
import pandas as pd

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
import itertools


from utils import number_split, create_mix, appendMetrics


from sklearn.manifold import TSNE


def do_Inference_HiddenSpece(txts):
    df_in = tokenizer(
        list(txts),
        return_tensors="pt",
        max_length=globalconfig.max_seq_length,
        padding="max_length",
        truncation=True,
    )

    ret_ls = []
    lst = list(range(len(df_in["input_ids"])))
    n = args.batch_size
    idx_ls = [lst[i : i + n] for i in range(len(lst)) if i % n == 0]

    model.eval()
    with torch.no_grad():
        for idx in tqdm(idx_ls, file=open(log_f, "w")):
            ret_output = model.roberta(
                input_ids=df_in["input_ids"][idx].to(globalconfig.device),
                attention_mask=df_in["attention_mask"][idx].to(globalconfig.device),
                output_hidden_states=True,
                return_dict=True,
                output_attentions=True,
            )

            ret_ls.append(ret_output[0][:, 0, :].detach().cpu().numpy())

    return np.concatenate(ret_ls)


# # Load Data & Model

### Main

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="CD", help="CD or HateSpeech")
argparser.add_argument("--isample", type=int, help="which sample setting to use")
argparser.add_argument("--methodUsed", type=str, help="which method to use")

args_input = argparser.parse_args()

dataset_name = args_input.dataset
isample = args_input.isample
methodUsed = args_input.methodUsed

# # dataset_name = "CD"
# # dataset_name = "HateSpeech"

# # isample = 566
# # isample = 6621


# dataset_name = "SHAC"


# isample = 1152
# # isample = 11063


# # methodUsed = "noaug"
# # methodUsed = "MMD"
# methodUsed = "GDRO"
# # methodUsed = "GradientReverse"

if dataset_name == "SHAC" and methodUsed == "MMD":
    ep = 6
else:
    ep = 3

if dataset_name == "SHAC":
    n_test_eval = 800
else:
    n_test_eval = 1000

if methodUsed == "noaug":
    weightsEdited = f"/bime-munin/xiruod/roberta-base_{dataset_name}-FullFT-noaug/n200/set-{isample}-epoch{ep}/pytorch_model.bin"
else:
    weightsEdited = f"/bime-munin/xiruod/{methodUsed}/roberta-base_{dataset_name}/n200/set-{isample}-epoch{ep}/pytorch_model.bin"


log_f = "../log/test.log"


class Args:
    model_name = "roberta-base"
    weightsEdited = weightsEdited
    output_dir = "~/Downloads/tmp"
    gpu = "3"
    device = "cuda:0"
    batch_size = 8
    dataset = dataset_name


args = Args()


class train_config:
    def __init__(self):
        self.quantization: bool = False


# tmp = [x for x in args.weightsEdited.split("/") if "set-" in x]
# # of form like set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1-added.pth
# name_pre = tmp[0].split(".pth")[0]
# # 7, 13, 70
# outdir = args.output_dir
# name_general = f"runningInferenceOnly"
# log_f = f"../log/{name_general}.log"

# os.makedirs(outdir, exist_ok=True)

globalconfig = train_config()
globalconfig.model_name = args.model_name
globalconfig.max_seq_length = 512
globalconfig.device = args.device

##### Tokenizer
tokenizer = AutoTokenizer.from_pretrained(globalconfig.model_name, use_fast=False)

# if args.cpuOps:
#     load_device = "cpu"
#     load_state_device = "cpu"
# else:
#     load_device = globalconfig.device
#     load_state_device = globalconfig.device
load_device = globalconfig.device
load_state_device = globalconfig.device

##### Load Model and  Update using Edited Weights
model = AutoModelForSequenceClassification.from_pretrained(
    globalconfig.model_name,
    use_safetensors=False,
)

# this step cannot be ignored here...

if methodUsed in ["GradientReverse", "GradientReverse_ForDomainBed"]:
    tmp = torch.load(
        args.weightsEdited,
        map_location="cpu",
        # map_location=lambda storage, loc: storage,
    )

    key_ToRemove = [
        "classifierDomain.dense.weight",
        "classifierDomain.dense.bias",
        "classifierDomain.out_proj.weight",
        "classifierDomain.out_proj.bias",
    ]

    for _ in key_ToRemove:
        del tmp[_]
else:
    tmp = torch.load(
        args.weightsEdited,
        map_location="cpu",
        # map_location=lambda storage, loc: storage,
    )


# this step cannot be ignored here...
model.load_state_dict(tmp)


model = model.to(globalconfig.device)

print("###  Finished Loading...")

# if args.quantization:
#     bnb_quantization_config = BnbQuantizationConfig(
#         load_in_8bit=True, llm_int8_threshold=6
#     )
#     model = load_and_quantize_model(
#         model,
#         weights_location=args.weightsEdited,
#         bnb_quantization_config=bnb_quantization_config,
#         device_map=load_device,
#     )

# print("###  Finished Quantization...")


if dataset_name == "CD":
    df_all = load_cd()
    df_avh = df_all["avh"]
    df_r56 = df_all["r56"]

    txt_col = "text"

    z_Categories = ["avh", "r56"]

    domain_col = "dfSource"
    df_split_label = "label_binary"
    y_Categories = [0, 1]

    labels = ["Negative", "Positive"]

elif dataset_name == "HateSpeech":
    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()

    txt_col = "text"

    z_Categories = ["dynGen", "wsf"]
    df_split_label = "label_binary"
    y_Categories = [0, 1]

    domain_col = "dfSource"
    labels = ["Negative", "Positive"]

elif dataset_name == "SHAC":
    #     df_dynGen = load_HateSpeech_dynGen()
    #     df_wsf = load_HateSpeech_wsf()

    df_shac = load_process_SHAC(replaceNA="all")
    domain_col = "location"
    label = "Drug"

    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x[label] else 0, axis=1)
    df_shac["dfSource"] = df_shac[domain_col]

    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

    txt_col = "text"

    df_split_label = "label_binary"
    y_Categories = [0, 1]

    z_Categories = ["uw", "mimic"]

    labels = ["Negative", "Positive"]


# # Select a balance set


pick_C = isample
n_test = 200
label = "label_binary"

if dataset_name == "CD":
    df0 = df_avh
    df1 = df_r56
    c = CD_DICT[f"c_n{n_test}_{pick_C}"]

elif dataset_name == "HateSpeech":
    df0 = df_dynGen
    df1 = df_wsf

    c = HateSpeech_DICT[f"c_n{n_test}_{pick_C}"]

elif dataset_name == "SHAC":
    df0 = df_shac_uw
    df1 = df_shac_mimic

    c = SHAC_DICT[f"c_n{n_test}_{pick_C}"]


dfs_used = create_mix(
    df0=df0,
    df1=df1,
    target=label,
    setting=c,
    sample=False,
    # seed=random.randint(0,1000),
    seed=222,
)

df0 = df0[~df0[txt_col].isin(dfs_used["train"][txt_col])].reset_index(drop=True)
df0 = df0[~df0[txt_col].isin(dfs_used["test"][txt_col])].reset_index(drop=True)


df1 = df1[~df1[txt_col].isin(dfs_used["train"][txt_col])].reset_index(drop=True)
df1 = df1[~df1[txt_col].isin(dfs_used["test"][txt_col])].reset_index(drop=True)


balanced_setting = number_split(
    p_pos_train_z0=0.3,
    p_pos_train_z1=0.3,
    p_mix_z1=0.5,
    alpha_test=1,
    train_test_ratio=1,
    n_test=n_test_eval,
    verbose=False,
)


balanced_setting


c = balanced_setting.copy()
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


dfs["test"].groupby("dfSource")[label].sum()


dfs["test"].groupby("dfSource").size()


dfs["test"]


# # Do Inference


array_hidden = do_Inference_HiddenSpece(dfs["test"][txt_col])


array_hidden.shape


# ## Fisher Ratio


def _one_fold_sw(Xa, Xb, n_projections=100, rng=None):
    """
    Compute sliced-Wasserstein (mean over projections) between two samples Xa, Xb.
    """
    dims = Xa.shape[1]
    vals = []
    for _ in range(n_projections):
        u = rng.normal(size=(dims,))
        u /= np.linalg.norm(u) + 1e-16
        pa = Xa.dot(u)
        pb = Xb.dot(u)
        vals.append(wasserstein_distance(pa, pb))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std()), vals


def sliced_wasserstein(
    X, z_Categories, domain_col=None, n_projections=100, random_state=0, n_splits=None
):
    """
    Compute sliced-Wasserstein separability for embeddings X with provenance labels.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Embeddings/features.
    z_Categories : array-like or DataFrame
        Provenance/domain labels (strings or ints) or DataFrame if domain_col given.
    domain_col : str or None
        If z_Categories is a DataFrame, the column name containing labels.
    n_projections : int
        Number of random 1-D projections to average over (default 100).
    random_state : int
        RNG seed.
    n_splits : int or None
        If int, perform StratifiedKFold with this many folds and compute SW per fold.
        If None (default), compute on the full dataset.

    Returns
    -------
    dict with keys:
      - "mean_sw": float, mean sliced-Wasserstein (averaged across projections and across folds if CV)
      - "std_sw": float, std across projections (and folds combined)
      - "per_projection_values": np.ndarray of shape (n_projections,) when n_splits is None,
                                 or shape (n_splits, n_projections) when CV used
      - "n_projections": int
      - "classes": list of class labels (string)
      - "fold_sw_means": list of per-fold mean SW (if n_splits provided)
      - "pairwise": dict mapping pair -> (mean, std) when >2 classes
    """
    # Extract labels
    if domain_col is not None:
        y_raw = z_Categories[domain_col].values
    else:
        y_raw = np.array(z_Categories)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_labels = le.classes_.tolist()
    unique = np.unique(y)
    rng = np.random.RandomState(random_state)

    # Helper: compute mean SW between two index sets
    def pair_sw(indices_a, indices_b):
        Xa = X[indices_a]
        Xb = X[indices_b]
        mean_val, std_val, vals = _one_fold_sw(
            Xa, Xb, n_projections=n_projections, rng=rng
        )
        return mean_val, std_val, vals

    # If no CV: compute on full set
    if n_splits is None:
        if len(unique) == 2:
            idx_a = np.where(y == unique[0])[0]
            idx_b = np.where(y == unique[1])[0]
            mean_val, std_val, vals = pair_sw(idx_a, idx_b)
            return {
                "mean_sw": mean_val,
                "std_sw": std_val,
                "per_projection_values": vals,
                "n_projections": n_projections,
                "classes": class_labels,
                "fold_sw_means": [mean_val],
                "pairwise": {(class_labels[0], class_labels[1]): (mean_val, std_val)},
            }
        else:
            # multi-class: average pairwise SW
            pairs = list(combinations(unique, 2))
            pairwise = {}
            all_vals = []
            means = []
            for i, j in pairs:
                idx_i = np.where(y == i)[0]
                idx_j = np.where(y == j)[0]
                m, s, vals = pair_sw(idx_i, idx_j)
                pairwise[(class_labels[i], class_labels[j])] = (m, s)
                all_vals.append(vals)
                means.append(m)
            all_vals = np.vstack(all_vals)  # shape (num_pairs, n_projections)
            # average across pairs then projections
            mean_sw = float(all_vals.mean())
            std_sw = float(all_vals.std())
            return {
                "mean_sw": mean_sw,
                "std_sw": std_sw,
                "per_projection_values": all_vals,  # pair x projection
                "n_projections": n_projections,
                "classes": class_labels,
                "fold_sw_means": [mean_sw],
                "pairwise": pairwise,
            }

    # If CV requested: compute per-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_vals = []
    fold_means = []
    pairwise_all_folds = []  # list of dicts per fold if multi-class

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        # compute SW on this fold's validation set (or you can choose train split; here val split)
        X_fold = X[val_idx]
        y_fold = y[val_idx]
        classes_fold = np.unique(y_fold)

        if len(classes_fold) < 2:
            # can't compute pairwise on this fold (rare if very imbalanced)
            fold_vals.append(np.zeros(n_projections))
            fold_means.append(0.0)
            pairwise_all_folds.append({})
            continue

        if len(classes_fold) == 2:
            ia = np.where(y_fold == classes_fold[0])[0]
            ib = np.where(y_fold == classes_fold[1])[0]
            # convert ia,ib to indices relative to X_fold and then map to global indices val_idx
            # we already have X_fold so use direct
            Xa = X_fold[ia]
            Xb = X_fold[ib]
            mean_val, std_val, vals = _one_fold_sw(
                Xa, Xb, n_projections=n_projections, rng=rng
            )
            fold_vals.append(vals)
            fold_means.append(mean_val)
            pairwise_all_folds.append(
                {
                    (class_labels[classes_fold[0]], class_labels[classes_fold[1]]): (
                        mean_val,
                        std_val,
                    )
                }
            )
        else:
            # multi-class: mean pairwise on this fold
            pairs = list(combinations(classes_fold, 2))
            all_pair_vals = []
            pairwise_fold = {}
            for i, j in pairs:
                ia = np.where(y_fold == i)[0]
                ib = np.where(y_fold == j)[0]
                Xa = X_fold[ia]
                Xb = X_fold[ib]
                m, s, vals = _one_fold_sw(Xa, Xb, n_projections=n_projections, rng=rng)
                pairwise_fold[(class_labels[i], class_labels[j])] = (m, s)
                all_pair_vals.append(vals)
            all_pair_vals = np.vstack(all_pair_vals)
            fold_vals.append(
                all_pair_vals.mean(axis=0)
            )  # fold-level projection values (averaged over pairs)
            fold_means.append(float(all_pair_vals.mean()))
            pairwise_all_folds.append(pairwise_fold)

    fold_vals = np.vstack(fold_vals)  # (n_splits, n_projections)
    mean_sw_overall = float(fold_vals.mean())
    std_sw_overall = float(fold_vals.std())

    return {
        "mean_sw": mean_sw_overall,
        "std_sw": std_sw_overall,
        "per_projection_values": fold_vals,
        "n_projections": n_projections,
        "classes": class_labels,
        "fold_sw_means": [float(m) for m in fold_means],
        "pairwise": pairwise_all_folds,
    }


res = sliced_wasserstein(
    array_hidden,
    z_Categories=dfs["test"][domain_col],
    n_projections=200,
    random_state=0,
)


print("\n ### Results ###")
print(dataset_name)

print(isample)

print(methodUsed)


print("Mean SW:", round(res["mean_sw"], 4), "Std SW:", round(res["std_sw"], 4))

print("\n\n")
