import os
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


### Temporary Argparse
import argparse

parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-c", "--CombinationIdx", type=int, help = "Set idx of c to use")
parser.add_argument("-q", "--quantization", action='store_true')

# Read arguments from command line
args = parser.parse_args()


class train_config:
    def __init__(self):
        self.quantization: bool = False

    
globalconfig = train_config()
globalconfig.quantization = args.quantization
globalconfig.device = "cuda:0"
globalconfig.profiler = False
# globalconfig.output_dir = "/bime-munin/xiruod/tmp/test_epoch3-llama-output-1244"
globalconfig.output_dir = f"/bime-munin/xiruod/tmp/alpha_train_0_8/quantization_epoch3-llama-output-{args.CombinationIdx}"
globalconfig.model_id = "/bime-munin/llama2_hf/llama-2-7b_hf/"
globalconfig.max_seq_length = 1024
globalconfig.num_train_epochs = 3
globalconfig.runs = 1    
globalconfig.lr = 1e-4
globalconfig.warmup_ratio = 0.1
######  Load Data

### Hate Speech
z_category = ['nothate', 'hate']

label2id = {z:idx for idx,z in zip(range(len(z_category)), z_category)}
id2label = {idx:z for idx,z in zip(range(len(z_category)), z_category)}

# (1) dynGen
df_dynGen = pd.read_csv("/bime-munin/xiruod/data/hateSpeech_Bulla2023/Dynamically-Generated-Hate-Speech-Dataset/Dynamically Generated Hate Dataset v0.2.3.csv",)

df_dynGen['label'] = df_dynGen['label'].map({"hate":"hate", "nothate":"nothate"})
df_dynGen["dfSource"] = "dynGen"
df_dynGen['label_binary'] = df_dynGen['label'].map({"hate":1,"nothate":0})

# (2)  wsf
ls_allFiles = pathlib.Path("/bime-munin/xiruod/data/hateSpeech_Bulla2023/hate-speech-dataset/all_files/").glob("*.txt")

ls_id = []
ls_text = []

for ifile in ls_allFiles:
    ls_id.append(ifile.name.split(".txt")[0])
    with open(ifile, "r") as f:
        ls_text.append(f.read())

df_wsf_raw = pd.DataFrame({"file_id":ls_id, "text":ls_text})

df_wsf_annotation = pd.read_csv("/bime-munin/xiruod/data/hateSpeech_Bulla2023/hate-speech-dataset/annotations_metadata.csv")

df_wsf = df_wsf_raw.merge(df_wsf_annotation, on="file_id", how="inner")

df_wsf = df_wsf[df_wsf['label'].isin(['hate','noHate'])].reset_index(drop=True)

# df_wsf['label_binary'] = df_wsf['label'].map({"hate":1,"noHate":0})
df_wsf['label'] = df_wsf['label'].map({"hate":"hate","noHate":"nothate"})
df_wsf["dfSource"] = "wsf"
df_wsf['label_binary'] = df_wsf['label'].map({"hate":1,"nothate":0})

##### Split
# Hate Speech Detection: df_dynGen (0.55) vs df_wsf (0.11)
n_test = 1000
train_test_ratio = 4


p_pos_train_z0_ls = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9] # probability of training set examples drawn from site/domain z0 being positive
p_pos_train_z1_ls = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9] # probability of test set examples drawn from site/domain z1 being positive

p_mix_z1_ls     = [0.2, 0.4, 0.6, 0.8] # = np.arange(0.1, 0.9, 0.05) 

# alpha_test_ls = np.arange(0, 10, 0.05)

numvals = 1023
base = 1.1
alpha_test_ls = np.power(base, np.arange(numvals))/np.power(base,numvals//2)


valid_full_settings = []
for combination in itertools.product(p_pos_train_z0_ls, 
                                     p_pos_train_z1_ls, 
                                     p_mix_z1_ls,
                                     alpha_test_ls
                                    ):
    

    number_setting = number_split(p_pos_train_z0=combination[0], 
                           p_pos_train_z1 = combination[1], 
                           p_mix_z1 = combination[2], alpha_test = combination[3],
                           train_test_ratio = train_test_ratio, 
                           n_test=n_test,
                                  verbose=False
                                 )

    if (number_setting is not None):
        if np.all([number_setting[k] >= 10 for k in list(number_setting.keys())[:-1]]):
            valid_full_settings.append(number_setting)
    
# run for check valid settings

import warnings; warnings.simplefilter('ignore')

# Validate settings
label='label_binary'
df0 = df_dynGen
df1 = df_wsf


valid_n_full_settings = []

for c in tqdm(valid_full_settings):
        c = c.copy()
        # create train/test split according to stats
        dfs = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, 
                         seed=222
                        )

        if dfs is None:
            continue
        
        valid_n_full_settings.append(c)

##### Tokenizer 
tokenizer = LlamaTokenizer.from_pretrained(globalconfig.model_id)

tokenizer.add_special_tokens({"pad_token":"<pad>"}) 



##### Dataset Loader and Tokenizer
def preprocess_function(examples):
    # tokenize
    ret = tokenizer(examples['text'], return_tensors='pt', max_length=globalconfig.max_seq_length, padding='max_length', truncation=True).to(globalconfig.device)

    return  ret

def datasets_loader(df):
    # from pandas df to Dataset & tokenize
    ret_datasets = datasets.Dataset.from_pandas(df[['text','dfSource','label_binary']].rename(columns={"label_binary":"label"}).reset_index(drop=True))
    ret_tokenized = ret_datasets.map(preprocess_function, batched=True)

    return ret_tokenized

##### Experiment - ONLY One Setting
pick_C = args.CombinationIdx

c = valid_n_full_settings[pick_C]

dfs = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, 
                 # seed=random.randint(0,1000),
                 seed=222
                )

tokenized_train = datasets_loader(dfs['train'])
tokenized_test = datasets_loader(dfs['test'])

## Define metric
def compute_metrics_twoLevels(eval_pred):
    # compute AUPRC, based on only two levels of Y
    predictions, labels = eval_pred
    probabilities = nn.functional.softmax(torch.FloatTensor(predictions), dim=-1)[:,1]

    auprc = average_precision_score(y_true=labels, y_score=probabilities)

    return {"auprc":auprc}

## Initialize model
model = LlamaForSequenceClassification.from_pretrained(globalconfig.model_id, 
                                         load_in_8bit=globalconfig.quantization, 
                                         device_map=globalconfig.device, 
                                                       torch_dtype=torch.float16 if globalconfig.quantization else torch.float32,
                                                       num_labels = len(id2label), 
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
        r=8,
        bias="none",
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"],
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
    'lora_config': lora_config,
    'learning_rate': globalconfig.lr,
    'num_train_epochs': globalconfig.num_train_epochs,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'gradient_checkpointing': False,
    'warmup_ratio':globalconfig.warmup_ratio,
}

# Set up profiler
if enable_profiler:
    # wait, warmup, active, repeat = 1, 1, 2, 1
    wait, warmup, active, repeat = 10, 10, 100, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    
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

    **{k:v for k,v in config.items() if k != 'lora_config'}
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
