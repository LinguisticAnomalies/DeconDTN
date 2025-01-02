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
parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
# parser.add_argument(
#     "-q", "--quantization", action="store_true", help="whether to use quantization"
# )
parser.add_argument("--model_name", default="", help="Model to use. Default RoBERTa")
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
parser.add_argument(
    "--GradientReverse",
    type=bool,
    default=False,
    help="If used Gradient Reverse Model",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import sys

sys.path.append("../src")
sys.path.append("../config")

from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC
from process_CD import load_cd
from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

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


class train_config:
    def __init__(self):
        self.quantization: bool = False


tmp = [x for x in args.weightsEdited.split("/") if "set-" in x]
# of form like set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1-added.pth
name_pre = tmp[0].split(".pth")[0]
# 7, 13, 70

name_general = f"runningInferenceOnly"
log_f = f"../log/{name_general}.log"


outdir = args.output_dir
os.makedirs(outdir, exist_ok=True)

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
if not args.GradientReverse:
    tmp = torch.load(
        args.weightsEdited,
        map_location="cpu",
        # map_location=lambda storage, loc: storage,
    )
else:
    tmp = torch.load(
        args.weightsEdited,
        map_location="cpu",
        # map_location=lambda storage, loc: storage,
    )

    key_ToRemove = ["classifierDomain.dense.weight", 
                    "classifierDomain.dense.bias", 
                    "classifierDomain.out_proj.weight", 
                    "classifierDomain.out_proj.bias",
                   ]

    for _ in key_ToRemove:
        del tmp[_]

# this step cannot be ignored here...
model.load_state_dict(
    tmp
)

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


def do_Inference(_df, txt_col):
    df_in = tokenizer(
        list(_df[txt_col]),
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
        for idx in tqdm(idx_ls, file=open(log_f, "w")):
            ret_output = model.forward(
                input_ids=df_in["input_ids"][idx].to(globalconfig.device),
                attention_mask=df_in["attention_mask"][idx].to(globalconfig.device),
            )
            y_probs_ = softmax(ret_output["logits"].cpu(), axis=1)
            y_ls.append(y_probs_)

        y_probs = np.concatenate(y_ls)

    n_cats = y_probs.shape[1]
    _df[["ycat_" + str(x) for x in range(n_cats)]] = y_probs
    return _df


######  Load Data & Save
if args.dataset == "SHAC":
    ### SHAC
    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x["Drug"] else 0, axis=1)

    txt_col = "text"

    df_shac["dfSource"] = df_shac["location"]
    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

    df_shac_uw = do_Inference(df_shac_uw, txt_col=txt_col)
    df_shac_uw.to_csv(f"{outdir}/inference_{name_pre}_df_shac_uw.csv")

    df_shac_mimic = do_Inference(df_shac_mimic, txt_col=txt_col)
    df_shac_mimic.to_csv(f"{outdir}/inference_{name_pre}_df_shac_mimic.csv")

elif args.dataset == "HateSpeech":

    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()

    txt_col = "text"

    df_dynGen = do_Inference(df_dynGen, txt_col=txt_col)
    df_dynGen.to_csv(f"{outdir}/inference_{name_pre}_df_dynGen.csv")

    df_wsf = do_Inference(df_wsf, txt_col=txt_col)
    df_wsf.to_csv(f"{outdir}/inference_{name_pre}_df_wsf.csv")

elif args.dataset == "CD":

    df_all = load_cd()
    df_avh = df_all["avh"]
    df_r56 = df_all["r56"]

    txt_col = "text"

    df_avh = do_Inference(df_avh, txt_col=txt_col)
    df_avh.to_csv(f"{outdir}/inference_{name_pre}_df_avh.csv")

    df_r56 = do_Inference(df_r56, txt_col=txt_col)
    df_r56.to_csv(f"{outdir}/inference_{name_pre}_df_r56.csv")
