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
    "--augMethod",
    default="noaug",
    choices=[
        "noaug",
        "EDA",
        "ReSample",
        "AllEqual-ReSample",
        "EDA_ReturnAll",
        "ReSample_by4",
    ],
    help="augup method",
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
    help="cuda device",
)
parser.add_argument(
    "--nTest",
    type=int,
    default=200,
    help="Number of testing samples",
)
parser.add_argument(
    "--num_train_epochs",
    type=int,
    default=3,
    help="Number of training epochs",
)
parser.add_argument("--batchSize", type=int, default=8, help="Batch size")
parser.add_argument(
    "--mntdir",
    type=str,
    default="/bime-munin/",
    help="Number of testing samples",
)
parser.add_argument("--reverseLabel", action="store_true")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import random
import sys
import itertools
from tqdm.auto import tqdm
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from copy import deepcopy

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

from utils import number_split, create_mix
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT

from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_SHAC import load_process_SHAC
from process_CD import load_cd
from eda import runEDA


class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.model_name = args.model_name
globalconfig.max_seq_length = 512
globalconfig.num_train_epochs = args.num_train_epochs  # 20 #3
globalconfig.runs = 1
globalconfig.lr = 1e-4
globalconfig.warmup_ratio = 0.1
globalconfig.profiler = False
globalconfig.device = args.device
globalconfig.per_device_train_batch_size = args.batchSize
globalconfig.per_device_eval_batch_size = args.batchSize
globalconfig.weight_decay = 1e-3
globalconfig.lr_scheduler_type = "constant"  # default "linear"

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


globalconfig.output_dir = f"{args.mntdir}/xiruod/{args.model_name}_{args.dataset}-FullFT-{args.augMethod}/n{args.nTest}/set-{args.CombinationIdx}-epoch{globalconfig.num_train_epochs}"

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


##### Tokenizer
tokenizer = AutoTokenizer.from_pretrained(globalconfig.model_name, use_fast=False)


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


##### Experiment - For AugUp

random.seed(12)
_rand = random.randint(0, 2**32 - 1)

df_training = deepcopy(dfs["train"])

if args.augMethod != "noaug":

    ###### Split for Augmentation and Remove..
    dfs_train_original = deepcopy(df_training)
    df0_train_pos = df_training.query(
        f"(`{df_split_label}`==True) and ({domain_col} == '{z_category[0]}')"
    )
    df0_train_neg = df_training.query(
        f"(`{df_split_label}`==False) and ({domain_col} == '{z_category[0]}')"
    )
    df1_train_pos = df_training.query(
        f"(`{df_split_label}`==True) and ({domain_col} == '{z_category[1]}')"
    )
    df1_train_neg = df_training.query(
        f"(`{df_split_label}`==False) and ({domain_col} == '{z_category[1]}')"
    )

    if args.augMethod == "AllEqual-ReSample":
        p_pos_train_target = 0.5
        c["mix_param_dict"]["p_mix_z1"] = 0.5
    else:
        p_pos_train_target = max(
            c["mix_param_dict"]["p_pos_train_z0"],
            c["mix_param_dict"]["p_pos_train_z1"],
        )

    for _t in np.arange(0, 1, 0.01):
        c_train_target = number_split(
            p_pos_train_z0=p_pos_train_target,
            p_pos_train_z1=p_pos_train_target,
            p_mix_z1=c["mix_param_dict"]["p_mix_z1"],
            alpha_test=_t,  # c['mix_param_dict']['alpha_test'],
            train_test_ratio=train_test_ratio,
            n_test=int(n_test),
            verbose=False,
        )

    def augDFReSample(df, name_to_aug):
        n_augs = c_train_target[name_to_aug] - c[name_to_aug]

        if n_augs < 0:
            df = df.sample(
                n=c_train_target[name_to_aug],
                random_state=_rand,
                replace=False,
                ignore_index=True,
            )
        elif n_augs > 0:

            _df = df.sample(
                n=n_augs,
                random_state=_rand,
                replace=True,
                ignore_index=True,
            )

            df = pd.concat([df, _df], ignore_index=True).reset_index(drop=True)

        return df

    def augDFReSample_by4(df, name_to_aug):
        n_augs = c[name_to_aug] * 4

        _df = df.sample(
            n=n_augs,
            random_state=_rand,
            replace=True,
            ignore_index=True,
        )

        df = pd.concat([df, _df], ignore_index=True).reset_index(drop=True)

        return df

    def augDFEDA(df, name_to_aug):
        n_augs = c_train_target[name_to_aug] - c[name_to_aug]

        if n_augs < 0:
            df = df.sample(
                n=c_train_target[name_to_aug],
                random_state=_rand,
                replace=False,
                ignore_index=True,
            )
        elif n_augs > 0:

            output_dir_eda = (
                f"/home/NETID/xiruod/projects/DeconDTN/output/eda/{args.dataset}"
            )
            os.makedirs(output_dir_eda, exist_ok=True)

            _df = runEDA(
                text_inputs=df[txt_col],
                file_tmp_for_eda=f"{output_dir_eda}/set-{args.CombinationIdx}.csv",
                file_eda_output=f"{output_dir_eda}/output-set-{args.CombinationIdx}.csv",
            )
            _df.rename(columns={"text": txt_col}, inplace=True)
            _df["label_binary"] = df["label_binary"].unique()[0]

            _df = _df.sample(
                n=n_augs,
                random_state=_rand,
                replace=True,
                ignore_index=True,
            )

            df = pd.concat([df, _df], ignore_index=True).reset_index(drop=True)

        return df

    def augDFEDA_ReturnAll(df, name_to_aug):

        output_dir_eda = (
            f"/home/NETID/xiruod/projects/DeconDTN/output/eda/{args.dataset}"
        )
        os.makedirs(output_dir_eda, exist_ok=True)

        _df = runEDA(
            text_inputs=df[txt_col],
            file_tmp_for_eda=f"{output_dir_eda}/set-{args.CombinationIdx}.csv",
            file_eda_output=f"{output_dir_eda}/output-set-{args.CombinationIdx}.csv",
        )
        _df.rename(columns={"text": txt_col}, inplace=True)
        _df["label_binary"] = df["label_binary"].unique()[0]

        df = pd.concat([df, _df], ignore_index=True).reset_index(drop=True)

        return df

    # if args.augMethod == "mixup":
    #     augDF = augDFmixup
    # elif args.augMethod == "ReSample":
    #     augDF = augDFReSample
    # elif args.augMethod == "LLM_Generate":
    #     augDF = augDFLLMGenerate
    if args.augMethod in ["ReSample", "AllEqual-ReSample"]:
        augDF = augDFReSample
    elif args.augMethod == "EDA":
        augDF = augDFEDA
    elif args.augMethod == "EDA_ReturnAll":
        augDF = augDFEDA_ReturnAll
    elif args.augMethod == "ReSample_by4":
        augDF = augDFReSample_by4

    df0_train_pos = augDF(df0_train_pos, "n_z0_pos_train")
    df0_train_neg = augDF(df0_train_neg, "n_z0_neg_train")
    df1_train_pos = augDF(df1_train_pos, "n_z1_pos_train")
    df1_train_neg = augDF(df1_train_neg, "n_z1_neg_train")

    df_training = pd.concat(
        [df0_train_pos, df0_train_neg, df1_train_pos, df1_train_neg],
        ignore_index=True,
    ).reset_index(drop=True)

    ####### Add a few columns in c
    c["mix_param_dict"]["p_pos_train_z0_new"] = p_pos_train_target
    c["mix_param_dict"]["p_pos_train_z1_new"] = p_pos_train_target
    c["mix_param_dict"]["C_y_train"] = (
        c["mix_param_dict"]["p_pos_train_z0_new"] * c["mix_param_dict"]["p_mix_z0"]
        + c["mix_param_dict"]["p_pos_train_z1_new"] * c["mix_param_dict"]["p_mix_z1"]
    )


tokenized_train = datasets_loader(df_training)
tokenized_test = datasets_loader(dfs["test"])


## Define metric
def compute_metrics_twoLevels(eval_pred):
    # compute AUPRC, based on only two levels of Y
    predictions, labels = eval_pred
    probabilities = nn.functional.softmax(torch.FloatTensor(predictions), dim=-1)[:, 1]

    auprc = average_precision_score(y_true=labels, y_score=probabilities)

    return {"auprc": auprc}


## Initialize model
torch.manual_seed(222)
torch.cuda.manual_seed(222)
torch.cuda.manual_seed_all(222)


model = AutoModelForSequenceClassification.from_pretrained(
    globalconfig.model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    use_safetensors=False,
)


model.train()


## Profiler

enable_profiler = globalconfig.profiler
output_dir = globalconfig.output_dir

config = {
    "learning_rate": globalconfig.lr,
    "num_train_epochs": globalconfig.num_train_epochs,
    "gradient_accumulation_steps": 2,
    "per_device_train_batch_size": globalconfig.per_device_train_batch_size,
    "per_device_eval_batch_size": globalconfig.per_device_eval_batch_size,
    "gradient_checkpointing": False,
    "warmup_ratio": globalconfig.warmup_ratio,
    "weight_decay": globalconfig.weight_decay,
    "lr_scheduler_type": globalconfig.lr_scheduler_type,
}

# Set up profiler
if enable_profiler:
    # wait, warmup, active, repeat = 1, 1, 2, 1
    wait, warmup, active, repeat = 10, 10, 100, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat
    )
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"{output_dir}/logs/tensorboard"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler

        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()


# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch",
    max_steps=total_steps if enable_profiler else -1,
    # **{k: v for k, v in config.items()},
    **config,
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_twoLevels,
        callbacks=[profiler_callback] if enable_profiler else [],
    )

    # Start training
    ret_train = trainer.train()
    ret_eval = trainer.evaluate()

# save metrics
ret = c
ret.update(ret_eval)
ret.update(ret_train.metrics)
trainer.save_metrics(split="all", metrics=ret)

ret_code = 1

model.save_pretrained(output_dir)
