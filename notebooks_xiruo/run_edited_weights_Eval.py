import os
import argparse

### Argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="SHAC",
    help="Dataset for the experiment",
)
# parser.add_argument("--weightsEdited", type=str, help="Path to edited weights")
parser.add_argument("--inferencePathPrefix", type=str, help="Path to inference file")
parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
parser.add_argument("--nRuns", type=int, default=1, help="Number of experiments to run")
parser.add_argument("--nTest", type=int, default=None, help="Size of testing set")
# parser.add_argument(
#     "--percent", type=int, default=5, help="X% of total setting will be used"
# )
# parser.add_argument(
#     "--sampleValidSettings",
#     action="store_true",
#     help="whether to sample valid settings",
# )
parser.add_argument(
    "--mntdir",
    type=str,
    default="/bime-munin/",
    help="Number of testing samples",
)
args = parser.parse_args()


import sys

sys.path.append("../src")
sys.path.append("../config")

from utils import number_split, create_mix
from sampling_numbers import HateSpeech_DICT, SHAC_DICT

from pathlib import Path
import itertools
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import random
from sklearn import metrics
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import warnings

warnings.simplefilter("ignore")


class train_config:
    def __init__(self):
        self.quantization: bool = False


######  Set Config Parameters
if args.nTest is None:
    n_test = int(
        [x for x in args.inferencePathPrefix.split("/") if x.startswith("n")][0].strip(
            "n"
        )
    )
else:
    n_test = args.nTest


runs = args.nRuns

# of form like inference_set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1-added
name_pre = Path(args.inferencePathPrefix).name.split("inference_")[1]
model_size = int([x for x in name_pre.split("-") if "B" in x][0].replace("B", ""))
assert model_size in (7, 13, 70)

outdir = args.output_dir

name_general = f"Eval-{name_pre}-ntest_{n_test}-Runs_{runs}"
log_f = f"../log/{name_general}.log"
_name_split = name_pre.split("-")
## set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth
pick_C = int(_name_split[_name_split.index("set") + 1])


globalconfig = train_config()
globalconfig.model_id = f"{args.mntdir}/llama2_hf/llama-2-{model_size}b_hf/"
globalconfig.max_seq_length = 1024

y_Categories = [0, 1]
n_yCats = len(y_Categories)


############  Load Data
train_test_ratio = 4

if args.dataset == "SHAC":
    df_shac_uw = pd.read_csv(f"{args.inferencePathPrefix}_df_shac_uw.csv")
    df_shac_mimic = pd.read_csv(f"{args.inferencePathPrefix}_df_shac_mimic.csv")

    p_pos_train_z0_ls = SHAC_DICT["Run-0"]["p_pos_train_z0_ls"]
    p_pos_train_z1_ls = SHAC_DICT["Run-0"]["p_pos_train_z1_ls"]
    p_mix_z1_ls = SHAC_DICT["Run-0"]["p_mix_z1_ls"]

    z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
    label = "label_binary"
    split_label = "Drug"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "location"
    df0 = df_shac_uw
    df1 = df_shac_mimic

    c = SHAC_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_2800"

elif args.dataset == "HateSpeech":
    df_dynGen = pd.read_csv(f"{args.inferencePathPrefix}_df_dynGen.csv")
    df_wsf = pd.read_csv(f"{args.inferencePathPrefix}_df_wsf.csv")

    p_pos_train_z0_ls = HateSpeech_DICT["Run-2"]["p_pos_train_z0_ls"]
    p_pos_train_z1_ls = HateSpeech_DICT["Run-2"]["p_pos_train_z1_ls"]
    p_mix_z1_ls = HateSpeech_DICT["Run-2"]["p_mix_z1_ls"]

    z_Categories = [
        "dynGen",
        "wsf",
    ]  # the order here matters! Should match with df0, df1
    label = "label_binary"
    split_label = "label_binary"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "dfSource"
    df0 = df_dynGen
    df1 = df_wsf

    c = HateSpeech_DICT[f"c_n{n_test}_{pick_C}"]


############ Diff Out Dataset used in LoRA Fine-Tuning
dfs_used = create_mix(
    df0=df0,
    df1=df1,
    target=label,
    setting=c,
    sample=False,
    # seed=random.randint(0,1000),
    seed=222,
)

assert dfs_used is not None

df0 = df0[~df0[txt_col].isin(dfs_used["train"][txt_col])].reset_index(drop=True)
df0 = df0[~df0[txt_col].isin(dfs_used["test"][txt_col])].reset_index(drop=True)


df1 = df1[~df1[txt_col].isin(dfs_used["train"][txt_col])].reset_index(drop=True)
df1 = df1[~df1[txt_col].isin(dfs_used["test"][txt_col])].reset_index(drop=True)


############  Get Split Configs & Further Limit Sampling Set, if necessary

# numvals = 1023
# base = 1.1
# alpha_test_ls = np.power(base, np.arange(numvals)) / np.power(base, numvals // 2)
alpha_test_ls = np.concatenate(
    [np.float_power(10, np.linspace(start=-2, stop=2, num=40)), [0.2, 1, 5]]
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

# # Select top 5 C_y's by 0.1 level
# dfgrp = tmp_df.groupby(["C_y", "C_y1"], as_index=False).size()
# dfgrp = dfgrp.sort_values("size", ascending=False)
# dfgrp = dfgrp.groupby("C_y1", as_index=False).head(1).iloc[:5, :]

# tmp_df = tmp_df[tmp_df["C_y"].isin(dfgrp["C_y"])]

# # Select top 5 p_mix_z1 by 0.1 level
# dfgrp = tmp_df.groupby(["p_mix_z1"], as_index=False).size()
# dfgrp = dfgrp.sort_values("size", ascending=False).iloc[:5, :]

# tmp_df = tmp_df[tmp_df["p_mix_z1"].isin(dfgrp["p_mix_z1"])]

# _sizegrp = (
#     tmp_df.groupby("C_y", as_index=False).size().sort_values("size", ascending=True)
# )
# cy_notsample = _sizegrp.iloc[:2, :]["C_y"].tolist()

# grouped = tmp_df.groupby(["C_y", "p_mix_z1"])

# _tmp = []
# for g, _dt in grouped:
#     _dt = _dt.sort_values("alpha_test")
#     _d_len = len(_dt.sort_values("alpha_test"))

#     if g[0] not in cy_notsample:
#         sampler = np.random.choice(
#             _d_len, int(args.percent / 100 * _d_len), replace=False
#         )
#         _tmp.append(_dt.iloc[sampler, :])
#     else:
#         sampler = np.random.choice(
#             _d_len, min(int((args.percent + 10) / 100 * _d_len), _d_len), replace=False
#         )
#         _tmp.append(_dt.iloc[sampler, :])

# tmp_df = pd.concat(_tmp)

valid_full_settings = tmp_df["combination"]


############  Eval

os.makedirs(outdir, exist_ok=True)

random.seed(123)
auprc_weightsEdited = []
auprc_weightsEdited_df0 = []
auprc_weightsEdited_df1 = []
auroc_weightsEdited = []
auroc_weightsEdited_df0 = []
auroc_weightsEdited_df1 = []

record_valid_settings_n = []


precision_weightsEdited = []
recall_weightsEdited = []
f1_weightsEdited = []
precision_weightsEdited_df0 = []
recall_weightsEdited_df0 = []
f1_weightsEdited_df0 = []
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


for iRun in range(runs):
    _rand = random.randint(0, 2**32 - 1)
    _n_setting = 0

    print(_rand)

    print(iRun)
    for c in tqdm(valid_full_settings, file=open(log_f, "w")):

        c = c.copy()

        dfs = create_mix(
            df0=df0,
            df1=df1,
            target=label,
            setting=c,
            sample=False,
            seed=_rand,
        )

        if dfs is None:
            continue

        # ##### NTOE: for shorter version!!!
        # if args.sampleValidSettings:
        #     if round(c['mix_param_dict']['alpha_train'], 4) not in [1, 2, 0.5, 4, 0.25, 6, 0.1667]:
        #         continue

        # _n_setting += 1
        # if _n_setting % args.percent != 0:
        #     continue

        c["run"] = iRun
        record_valid_settings_n.append(c)

        y_train = dfs["train"][label]
        y_test = dfs["test"][label]

        n_test = len(y_test)

        y_probs_auprc_weightsEdited = dfs["test"][["ycat_0", "ycat_1"]].values

        ret = c

        ret_code = 1

        auprc_weightsEdited.append(
            metrics.average_precision_score(
                y_true=y_test, y_score=y_probs_auprc_weightsEdited[:, 1]
            )
        )
        auprc_weightsEdited_df0.append(
            metrics.average_precision_score(
                y_true=y_test[dfs["test"][domain_col] == z_Categories[0]],
                y_score=y_probs_auprc_weightsEdited[
                    dfs["test"][domain_col] == z_Categories[0], 1
                ],
            )
        )
        auprc_weightsEdited_df1.append(
            metrics.average_precision_score(
                y_true=y_test[dfs["test"][domain_col] == z_Categories[1]],
                y_score=y_probs_auprc_weightsEdited[
                    dfs["test"][domain_col] == z_Categories[1], 1
                ],
            )
        )
        auroc_weightsEdited.append(
            roc_auc_score(y_true=y_test, y_score=y_probs_auprc_weightsEdited[:, 1])
        )
        auroc_weightsEdited_df0.append(
            roc_auc_score(
                y_true=y_test[dfs["test"][domain_col] == z_Categories[0]],
                y_score=y_probs_auprc_weightsEdited[
                    dfs["test"][domain_col] == z_Categories[0], 1
                ],
            )
        )
        auroc_weightsEdited_df1.append(
            roc_auc_score(
                y_true=y_test[dfs["test"][domain_col] == z_Categories[1]],
                y_score=y_probs_auprc_weightsEdited[
                    dfs["test"][domain_col] == z_Categories[1], 1
                ],
            )
        )
        t = precision_recall_fscore_support(
            y_true=y_test,
            y_pred=y_probs_auprc_weightsEdited[:, 1] > 0.5,
            average="binary",
            pos_label=1,
        )
        t_df0 = precision_recall_fscore_support(
            y_true=y_test[dfs["test"][domain_col] == z_Categories[0]],
            y_pred=y_probs_auprc_weightsEdited[
                dfs["test"][domain_col] == z_Categories[0], 1
            ]
            > 0.5,
            average="binary",
            pos_label=1,
        )
        t_df1 = precision_recall_fscore_support(
            y_true=y_test[dfs["test"][domain_col] == z_Categories[1]],
            y_pred=y_probs_auprc_weightsEdited[
                dfs["test"][domain_col] == z_Categories[1], 1
            ]
            > 0.5,
            average="binary",
            pos_label=1,
        )
        precision_weightsEdited.append(t[0])
        recall_weightsEdited.append(t[1])
        f1_weightsEdited.append(t[2])
        precision_weightsEdited_df0.append(t_df0[0])
        recall_weightsEdited_df0.append(t_df0[1])
        f1_weightsEdited_df0.append(t_df0[2])
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
        "f1_weightsEdited_df0": f1_weightsEdited_df0,
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
        "auroc_weightsEdited": auroc_weightsEdited,
        "auroc_weightsEdited_df0": auroc_weightsEdited_df0,
        "auroc_weightsEdited_df1": auroc_weightsEdited_df1,
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
