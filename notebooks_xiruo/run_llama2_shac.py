import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

from utils import number_split, create_mix


from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC

### Temporary Argparse
import argparse

parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-c", "--CombinationIdx", type=int, help="Set idx of c to use")
parser.add_argument("-q", "--quantization", action="store_true")
parser.add_argument("--lora_r", type=int, default=8, help="Set LoRA r value")
parser.add_argument(
    "--model_size", type=int, default=7, help="Llama 2 size: 7, 13, or 70"
)

# Read arguments from command line
args = parser.parse_args()


class train_config:
    def __init__(self):
        self.quantization: bool = False

        


globalconfig = train_config()
globalconfig.quantization = args.quantization
globalconfig.model_id = f"/bime-munin/llama2_hf/llama-2-{args.model_size}b_hf/"
globalconfig.max_seq_length = 1024
globalconfig.num_train_epochs = 3
globalconfig.runs = 1
globalconfig.lr = 1e-4
globalconfig.warmup_ratio = 0.1
globalconfig.lora_r = args.lora_r
globalconfig.profiler = False
globalconfig.device = "cuda:0"

if args.model_size == 70:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    globalconfig.per_device_train_batch_size = 1 #2
    globalconfig.per_device_eval_batch_size = 1 #2
    

else:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    globalconfig.per_device_train_batch_size = 8
    globalconfig.per_device_eval_batch_size = 8
    

if args.quantization:
    dir_q_snippet = "quantization"
else:
    dir_q_snippet = "NOquantization"

globalconfig.output_dir = f"/bime-munin/xiruod/llama2_SHAC/n200/set-{args.CombinationIdx}-{dir_q_snippet}-epoch3-llama-2-{args.model_size}B-loraR-{args.lora_r}"
# globalconfig.output_dir = f"~/llama2_SHAC/n200/set-{args.CombinationIdx}-{dir_q_snippet}-epoch3-llama-2-{args.model_size}B-loraR-{args.lora_r}"


######  Load Data

### SHAC
label = "Drug"
z_category = ["uw", "mimic"]
y_cat = ["False", "True"]

label2id = {z: idx for idx, z in zip(range(len(y_cat)), y_cat)}
id2label = {idx: z for idx, z in zip(range(len(y_cat)), y_cat)}

df_shac = load_process_SHAC(replaceNA="all")

df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x[label] else 0, axis=1)
df_shac["dfSource"] = df_shac["location"]

df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

##### Split
# SHAC-Drug - Balanced Alpha
n_test = 200
train_test_ratio = 4


p_pos_train_z0_ls = np.arange(
    0, 1, 0.1
)  # probability of training set examples drawn from site/domain z0 being positive
p_pos_train_z1_ls = np.arange(
    0, 1, 0.1
)  # probability of test set examples drawn from site/domain z1 being positive

p_mix_z1_ls = np.arange(0, 1, 0.05)

numvals = 1023
base = 1.1
alpha_test_ls = np.power(base, np.arange(numvals)) / np.power(base, numvals // 2)

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


# run for check valid settings

import warnings

warnings.simplefilter("ignore")

# Validate settings

df0 = df_shac_uw
df1 = df_shac_mimic


valid_n_full_settings = []

for c in tqdm(valid_full_settings):
    c = c.copy()
    # create train/test split according to stats
    dfs = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, seed=222)

    if dfs is None:
        continue

    valid_n_full_settings.append(c)

##### Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(f"/bime-munin/llama2_hf/llama-2-7b_hf/")

tokenizer.add_special_tokens({"pad_token": "<pad>"})


##### Dataset Loader and Tokenizer
def preprocess_function(examples):
    # tokenize
    ret = tokenizer(
        examples["text"],
        return_tensors="pt",
        max_length=globalconfig.max_seq_length,
        padding="max_length",
        truncation=True,
    ).to(globalconfig.device)

    return ret


def datasets_loader(df):
    # from pandas df to Dataset & tokenize
    ret_datasets = datasets.Dataset.from_pandas(
        df[["text", "dfSource", "label_binary"]]
        .rename(columns={"label_binary": "label"})
        .reset_index(drop=True)
    )
    ret_tokenized = ret_datasets.map(preprocess_function, batched=True)

    return ret_tokenized


##### Experiment - ONLY One Setting
pick_C = args.CombinationIdx

c = valid_n_full_settings[pick_C]
print("Balanced? Check setting....")
print(c)
dfs = create_mix(
    df0=df0,
    df1=df1,
    target=label,
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
model = LlamaForSequenceClassification.from_pretrained(
    globalconfig.model_id,
    load_in_8bit=globalconfig.quantization,
    device_map=globalconfig.device,
    # device_map="auto",
    torch_dtype=torch.float16 if globalconfig.quantization else torch.float32,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
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
    bf16=globalconfig.quantization,  # Use BF16 if available
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
