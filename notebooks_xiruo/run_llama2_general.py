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
parser.add_argument("-q", "--quantization", action="store_true")
parser.add_argument("--lora_r", type=int, default=8, help="Set LoRA r value")
parser.add_argument(
    "--model_size", type=int, default=7, help="Llama 2 size: 7, 13, or 70"
)
parser.add_argument("--toPredict", default="Target", help="Target vs Source")
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
parser.add_argument("--reverseLabel", action="store_true")
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    LlamaTokenizer,
    LlamaForSequenceClassification,
    TrainerCallback,
    default_data_collator,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
)


sys.path.append("../src")
sys.path.append("../config")

from utils import number_split, create_mix
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT

from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_SHAC import load_process_SHAC
from process_CD import load_cd


class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.quantization = args.quantization
globalconfig.model_id = f"{args.mntdir}/llama2_hf/llama-2-{args.model_size}b_hf/"
globalconfig.max_seq_length = 1024
globalconfig.num_train_epochs = 3
globalconfig.runs = 1
globalconfig.lr = 1e-4
globalconfig.warmup_ratio = 0.1
globalconfig.lora_r = args.lora_r
globalconfig.profiler = False
globalconfig.device = args.device
globalconfig.per_device_train_batch_size = args.batchSize
globalconfig.per_device_eval_batch_size = args.batchSize


if args.quantization:
    dir_q_snippet = "quantization"
else:
    dir_q_snippet = "NOquantization"

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

if args.toPredict == "Target":
    globalconfig.output_dir = f"{args.mntdir}/xiruod/llama2_{args.dataset}/n{args.nTest}/set-{args.CombinationIdx}-{dir_q_snippet}-epoch{globalconfig.num_train_epochs}-llama-2-{args.model_size}B-loraR-{args.lora_r}"

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


elif args.toPredict == "Source":
    label = domain_col
    globalconfig.output_dir = f"{args.mntdir}/xiruod/llama2_{args.dataset}/n{args.nTest}/Source-set-{args.CombinationIdx}-{dir_q_snippet}-epoch{globalconfig.num_train_epochs}-llama-2-{args.model_size}B-loraR-{args.lora_r}"

    if args.reverseLabel:
        z_category.reverse()
        globalconfig.output_dir = f"{args.mntdir}/xiruod/llama2_{args.dataset}/n{args.nTest}/Reverse-Source-set-{args.CombinationIdx}-{dir_q_snippet}-epoch{globalconfig.num_train_epochs}-llama-2-{args.model_size}B-loraR-{args.lora_r}"

    label2id = {z: idx for idx, z in zip(range(len(z_category)), z_category)}
    id2label = {idx: z for idx, z in zip(range(len(z_category)), z_category)}

    if args.dataset == "SHAC":
        df_shac = load_process_SHAC(replaceNA="all")

        df_shac["label_binary"] = df_shac.apply(lambda x: label2id[x[label]], axis=1)
        df_shac["dfSource"] = df_shac[domain_col]

    elif args.dataset == "HateSpeech":
        df_dynGen = load_HateSpeech_dynGen()
        df_wsf = load_HateSpeech_wsf()

        df_dynGen.rename(columns={"label_binary": "target_binary"}, inplace=True)
        df_wsf.rename(columns={"label_binary": "target_binary"}, inplace=True)

        df_dynGen["label_binary"] = df_dynGen.apply(
            lambda x: label2id[x[label]], axis=1
        )
        df_wsf["label_binary"] = df_wsf.apply(lambda x: label2id[x[label]], axis=1)

    elif args.dataset == "CD":
        df_all = load_cd()
        df_avh = df_all["avh"]
        df_r56 = df_all["r56"]

        df_avh.rename(columns={"label_binary": "target_binary"}, inplace=True)
        df_r56.rename(columns={"label_binary": "target_binary"}, inplace=True)

        df_avh["label_binary"] = df_avh.apply(lambda x: label2id[x[label]], axis=1)
        df_r56["label_binary"] = df_r56.apply(lambda x: label2id[x[label]], axis=1)

else:
    sys.exit("Unknown Outcome: 'Target' and 'Source' ONLY")

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
    df_split_label = "label_binary" if args.toPredict == "Target" else "target_binary"

    c = HateSpeech_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n1000_9870"

elif args.dataset == "CD":
    df0 = df_avh
    df1 = df_r56
    df_split_label = "label_binary" if args.toPredict == "Target" else "target_binary"

    c = CD_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_566"


# run for check valid settings

import warnings

warnings.simplefilter("ignore")


##### Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(
    f"{args.mntdir}/llama2_hf/llama-2-7b_hf/", use_safetensors=False
)

tokenizer.add_special_tokens({"pad_token": "<pad>"})


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

tokenized_train = datasets_loader(dfs["train"])
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

model = LlamaForSequenceClassification.from_pretrained(
    globalconfig.model_id,
    load_in_8bit=globalconfig.quantization,
    # device_map=globalconfig.device,
    device_map="auto",
    # torch_dtype=torch.float16 if globalconfig.quantization else torch.float32,
    torch_dtype=torch.bfloat16 if globalconfig.quantization else torch.float32,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    use_safetensors=False,
)


model.config.pad_token_id = tokenizer.pad_token_id

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

model.train()


## Peft Config
def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=globalconfig.lora_r,
        bias="none",
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["classifier"],
    )

    # prepare int-8 model for training
    if globalconfig.quantization:
        model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


model, lora_config = create_peft_config(model)

## Profiler

enable_profiler = globalconfig.profiler
output_dir = globalconfig.output_dir

config = {
    "lora_config": lora_config,
    "learning_rate": globalconfig.lr,
    "num_train_epochs": globalconfig.num_train_epochs,
    "gradient_accumulation_steps": 2,
    "per_device_train_batch_size": globalconfig.per_device_train_batch_size,
    "per_device_eval_batch_size": globalconfig.per_device_eval_batch_size,
    "gradient_checkpointing": False,
    "warmup_ratio": globalconfig.warmup_ratio,
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
    # bf16=globalconfig.quantization,  # Use BF16 if available
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch_fused" if globalconfig.quantization else "adamw_torch",
    max_steps=total_steps if enable_profiler else -1,
    # max_steps=50,
    **{k: v for k, v in config.items() if k != "lora_config"},
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
