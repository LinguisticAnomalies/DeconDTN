import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import datasets
import random
from contextlib import nullcontext
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from torch import nn
from transformers import default_data_collator, Trainer, TrainingArguments

import itertools
from tqdm.auto import tqdm

import torch


sys.path.append("../src")

from utils import number_split, create_mix
from process_SHAC import load_process_SHAC


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
)


##### Dataset Loader and Tokenizer
def preprocess_function(examples):
    # tokenize
    ret = tokenizer(examples['text'], return_tensors='pt', max_length=globalconfig.max_seq_length, padding='max_length', truncation=True).to(globalconfig.device)

    return  ret

def datasets_loader(df, txt_col):
    # from pandas df to Dataset & tokenize
    ret_datasets = datasets.Dataset.from_pandas(df[[txt_col,"label"]].reset_index(drop=True))
    ret_tokenized = ret_datasets.map(preprocess_function, batched=True)

    return ret_tokenized

def create_peft_config(model):

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        bias="none",
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["query", "value"],
        modules_to_save=["classifier"],
    )

    # prepare int-8 model for training
    if globalconfig.quantization:
        model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

## Define metric
def compute_metrics_twoLevels(eval_pred):
    # compute AUPRC, based on only two levels of Y
    predictions, labels = eval_pred
    probabilities = nn.functional.softmax(torch.FloatTensor(predictions), dim=-1)[:,1]

    auprc = average_precision_score(y_true=labels, y_score=probabilities)

    return {"auprc":auprc}


####################
## Load Data - SHAC
####################

df_shac = load_process_SHAC(replaceNA="all")

z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
label='Drug'
n_zCats = len(z_Categories)
txt_col="text"
domain_col = "location"

y_cat = [0, 1]


# Create binary version of "label"
assert "label" not in df_shac.columns

df_shac['label'] = df_shac[label].astype(int)

df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

df0 = df_shac_uw
df1 = df_shac_mimic


label2id = {y:idx for idx,y in zip(range(len(y_cat)), y_cat)}
id2label = {idx:y for idx,y in zip(range(len(y_cat)), y_cat)}







##### Split
# SHAC-Drug - Balanced Alpha
## Only selecting C_y in [0.2, 0.48, 0.72]
n_test = 200
train_test_ratio = 4


p_pos_train_z0_ls = np.arange(0, 1, 0.1) # probability of training set examples drawn from site/domain z0 being positive
p_pos_train_z1_ls = np.arange(0, 1, 0.1) # probability of test set examples drawn from site/domain z1 being positive

p_mix_z1_ls     = np.arange(0, 1, 0.05) 

numvals = 129
base = 1.01

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

    if (number_setting is not None) and (number_setting['mix_param_dict']['C_y'] in [0.2, 0.48, 0.72]) and (number_setting['mix_param_dict']['alpha_train'] in [1., 3, 5, 1/3, 0.2]):
        if np.all([number_setting[k] >= 10 for k in list(number_setting.keys())[:-1]]):
            valid_full_settings.append(number_setting)
    
    
    
    
# run for check valid settings

import warnings; warnings.simplefilter('ignore')

# Validate settings

df0 = df_shac_uw
df1 = df_shac_mimic


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

    
    
#####################
## Model Setup
####################

class train_config:
    def __init__(self):
        self.quantization: bool = False

    
    
globalconfig = train_config()
globalconfig.model_id="bert-base-uncased"
globalconfig.quantization = False
globalconfig.device = "cuda:0"
globalconfig.runs = 3
globalconfig.profiler = False
globalconfig.max_seq_length=512
globalconfig.num_train_epochs=3
globalconfig.lr = 1e-4
globalconfig.warmup_ratio = 0.1

rand_seed_np = 24
rand_seed_torch = 187


###########################
###--------    RUN!!!
##########################

# run for check valid settings
random.seed(rand_seed_np)
np.random.seed(rand_seed_np)
torch.manual_seed(rand_seed_torch)
torch.cuda.manual_seed(rand_seed_torch)

# Validate settings

df0 = df_shac_uw
df1 = df_shac_mimic


valid_n_full_settings = []

tokenizer = AutoTokenizer.from_pretrained(globalconfig.model_id)

for ct,c in enumerate(tqdm(valid_full_settings)):

    c = c.copy()
    # create train/test split according to stats
    dfs = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, 
                     seed=222
                    )

    if dfs is None:
        continue
    
    valid_n_full_settings.append(c)
    
    
    for run_i in range(globalconfig.runs):
        dfs = create_mix(
                df0=df0,
                df1=df1,
                target=label,
                setting=c,
                sample=False,
                seed=random.randint(0, 1000),
            )

        assert dfs is not None
        
        # Init model
        model = AutoModelForSequenceClassification.from_pretrained(globalconfig.model_id)
        
        ## Peft Config
        model, lora_config = create_peft_config(model)
        
        ## Profiler

        
        globalconfig.output_dir = f"/gscratch/scrubbed/xiruod/LoRA_BERT_SHAC/exp_{ct}_run_{run_i}"
        
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
        profiler = nullcontext()
        
        tokenized_train = datasets_loader(dfs['train'], txt_col=txt_col)
        tokenized_test = datasets_loader(dfs['test'], txt_col=txt_col)
        
        # Define training args
        training_args = TrainingArguments(
            output_dir=globalconfig.output_dir,
            overwrite_output_dir=True,
            bf16=globalconfig.quantization,  # Use BF16 if available
            # logging strategies
            logging_dir=f"{globalconfig.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="no",
            optim="adamw_torch_fused" if globalconfig.quantization else "adamw_torch",
            max_steps=-1,
            
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
                callbacks=[],
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
        
        
        # model.save_pretrained(output_dir)