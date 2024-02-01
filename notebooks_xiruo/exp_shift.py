import sys
import os
import logging

sys.path.append("../src")


import pandas as pd
import numpy as np

import random
import itertools
from sklearn import metrics


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer



import math

from utils import number_split, create_mix
from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC
from custom_distance import KL



from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

import pickle


from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


from sentence_transformers import SentenceTransformer

from sklearn.metrics import precision_recall_fscore_support

from transformers import AutoTokenizer, AutoModel

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

from torch import nn
from transformers import (
    Trainer,
    TrainingArguments,
    LlamaTokenizer,
    LlamaForSequenceClassification,
    TrainerCallback,
    default_data_collator,
)
from torch.utils.data import DataLoader
from peft import (
        PeftConfig,
        PeftModel,
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
)


########
## Load Dataset
#######
import pathlib

df_dynGen = pd.read_csv("/bime-munin/xiruod/data/hateSpeech_Bulla2023/Dynamically-Generated-Hate-Speech-Dataset/Dynamically Generated Hate Dataset v0.2.3.csv",)
df_dynGen['label_binary'] = df_dynGen['label'].map({"hate":1, "nothate":0})



df_dynGen["dfSource"] = "dynGen"

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

df_wsf['label_binary'] = df_wsf['label'].map({"hate":1,"noHate":0})

df_wsf["dfSource"] = "wsf"





###### Split

# Hate Speech Detection: df_dynGen (0.55) vs df_wsf (0.11)
n_test = 1000
train_test_ratio = 4


p_pos_train_z0_ls = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9] # probability of training set examples drawn from site/domain z0 being positive
p_pos_train_z1_ls = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9] # probability of test set examples drawn from site/domain z1 being positive

p_mix_z1_ls     = [0.2, 0.4, 0.6, 0.8] # = np.arange(0.1, 0.9, 0.05) 

# alpha_test_ls = np.arange(0, 10, 0.05)

numvals = 513
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

label='label_binary'
df0 = df_dynGen
df1 = df_wsf

# label='Drug'
# df0 = df_shac_uw
# df1 = df_shac_mimic

# label='y_true'
# df0 = df_christian
# df1 = df_nonchristian
# df0 = df_white
# df1 = df_notwhite
# df0 = df_male
# df1 = df_notmale

valid_n_full_settings = []

for c in tqdm(valid_full_settings, desc="Checking valid settings"):
    # for c in test_settings:


        c = c.copy()
        # create train/test split according to stats
        # dfs = create_mix(df0=df_wls_merge, df1=df_adress, target='label', setting= c, sample=False)
        # dfs = create_mix(df0=df_shac_uw, df1=df_shac_mimic, target=label, setting= c, sample=False, seed=random.randint(0,1000))
        dfs = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, 
                         # seed=random.randint(0,1000),
                         seed=222
                        )

        if dfs is None:
            continue
        
        valid_n_full_settings.append(c)

####### Setup LoRA Model

class train_config:
    def __init__(self):
        self.quantization: bool = False

    
globalconfig = train_config()
globalconfig.quantization = True
globalconfig.device = "cuda:0"
globalconfig.profiler = False
globalconfig.output_dir = f"/bime-munin/xiruod/tmp/quantization_epoch3-llama-output-1244/"
globalconfig.model_id = "/bime-munin/llama2_hf/llama-2-7b_hf/"
globalconfig.max_seq_length = 1024
globalconfig.num_train_epochs = 3
globalconfig.runs = 1    
globalconfig.lr = 1e-4
globalconfig.warmup_ratio = 0.1


# peft_model_id = "/bime-munin/xiruod/tmp/llama-output/"
peft_model_dir = globalconfig.output_dir

config = PeftConfig.from_pretrained(peft_model_dir)

model = LlamaForSequenceClassification.from_pretrained(config.base_model_name_or_path, 
                                                       return_dict=True, 
                                                       load_in_8bit=globalconfig.quantization, 
                                                       device_map=globalconfig.device,
                                                      )

tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.add_special_tokens({"pad_token":"<pad>"}) 

model.config.pad_token_id = tokenizer.pad_token_id

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_dir)


softmax = torch.nn.Softmax(dim=1)


##### Dataset Loader and Tokenizer
def preprocess_function(examples):
    # tokenize
    ret = tokenizer(examples['text'], return_tensors='pt', max_length=globalconfig.max_seq_length, padding='max_length', truncation=True).to(globalconfig.device)

    return  ret

def datasets_loader(df):
    # from pandas df to Dataset & tokenize
    ret_datasets = datasets.Dataset.from_pandas(df[['text','dfSource','label_binary']].rename(columns={"label_binary":"label"}).reset_index(drop=True)).with_format("torch")
    ret_tokenized = ret_datasets.map(preprocess_function, batched=True)

    return ret_tokenized



#### Diff Out Training Dataset
pick_C = int(globalconfig.output_dir.split("/")[-2].split("-")[-1])

c = valid_n_full_settings[pick_C]

dfs = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, 
                 # seed=random.randint(0,1000),
                 seed=222
                )

df_dynGen = df_dynGen[~df_dynGen['text'].isin(dfs['train']['text'])].reset_index(drop=True)
df_dynGen = df_dynGen[~df_dynGen['text'].isin(dfs['test']['text'])].reset_index(drop=True)

df_wsf = df_wsf[~df_wsf['text'].isin(dfs['train']['text'])].reset_index(drop=True)
df_wsf = df_wsf[~df_wsf['text'].isin(dfs['test']['text'])].reset_index(drop=True)




###### Run Experiments
#### Crazy version....

import warnings; warnings.simplefilter('ignore')

##### Used for AMIA Paper
# transform = "Sentence-BERT"
# transform = "binaryUnigram"
transform = "LoRA"


runs = 5


### Hate Speech
z_Categories = ["dynGen", "wsf"]  # the order here matters! Should match with df0, df1
label='label_binary'
n_zCats = len(z_Categories)
txt_col="text"
domain_col = "dfSource"
df0 = df_dynGen
df1 = df_wsf
outdir = f"../output/regressionHateSpeechBalanceAlpha"


### SHAC
# z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
# label='Drug'
# n_zCats = len(z_Categories)
# txt_col="text"
# domain_col = "location"
# df0 = df_shac_uw
# df1 = df_shac_mimic
# outdir = f"../output/regressionSHACBalanceAlpha"

### CivilComments from WILDS, by christian
# z_Categories = ["christian", "nonchristian"]  # the order here matters! Should match with df0, df1
# label='y_true'
# n_zCats = len(z_Categories)
# txt_col="text"
# domain_col = "Christian"
# df0 = df_christian
# df1 = df_nonchristian
# outdir = f"../output/regressionCivilComments_by_Christian"

### CivilComments from WILDS, by White
# z_Categories = ["white", "notwhite"]  # the order here matters! Should match with df0, df1
# label='y_true'
# n_zCats = len(z_Categories)
# txt_col="text"
# domain_col = "White"
# df0 = df_white
# df1 = df_notwhite
# outdir = f"../output/regressionCivilComments_by_White"

### CivilComments from WILDS, by Male
# z_Categories = ["male", "notmale"]  # the order here matters! Should match with df0, df1
# label='y_true'
# n_zCats = len(z_Categories)
# txt_col="text"
# domain_col = "Male"
# df0 = df_male
# df1 = df_notmale
# outdir = f"../output/regressionCivilComments_by_Male"

# save to file
# outname = f"../output/regressionInverseSHAC_MIMIC_UW/{transform}_{p_pos_train_z0}_{p_pos_train_z1}_{n_test}_{penalty}_C{C}_V{v}.pkl"
# outname = f"../output/regressionSHAC/{transform}_{p_pos_train_z0}_{p_pos_train_z1}_{n_test}_{penalty}_C{C}_V{v}.pkl"

os.makedirs(outdir, exist_ok=True)

logging.basicConfig(filename=f'{outdir}/logs.txt')


##### Test for LLaMa
# transform = "LLaMaAverage"

# runs = 5

# model = SentenceTransformer('all-MiniLM-L6-v2')
# vectorizer = CountVectorizer(binary=True, min_df=1, stop_words='english')

# z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
# label='Drug'
# n_zCats = len(z_Categories)
# txt_col="LLaMaEmbeddings"
# domain_col = "location"
# df0 = df_shac_llama_average_uw
# df1 = df_shac_llama_average_mimic


### IMDB
# z_Categories = ["Horror","Documentary"]
# label='label_binary'
# n_zCats = len(z_Categories)
# z_Categories = ["Horror","nonHorror"]
# label='label_binary'
# n_zCats = len(z_Categories)

### Yelp by States, AZ vs MO
# z_Categories = ["AZ","MO"]
# label='label'
# n_zCats = len(z_Categories)
# txt_col = "text"
# domain_col = 'state'
# df0 = df_AZ
# df1 = df_MO

### Yelp by Year
# z_Categories = ["<=2015",">=2020"]
# label='label'
# n_zCats = len(z_Categories)
# txt_col = "text"
# domain_col = 'year_cut'
# df0 = df_before2015
# df1 = df_after2020

y_Categories = [0,1]
n_yCats = len(y_Categories)


# setting for logistic regression
# penalty = "l1"
# solver = "liblinear"
penalty = "l2"
solver = "lbfgs"



random.seed(123)
auprc_logistic_confounder = []
auprc_logistic_confounder_df0 = []
auprc_logistic_confounder_df1 = []

auprc_logistic_vanilla = []
auprc_logistic_vanilla_df0 = []
auprc_logistic_vanilla_df1 = []

valid_n_full_settings = []

# [[1,10], [1,1],[1,100]]
for C, v in [[1,10], ]:
    auprc_logistic_confounder = []
    auprc_logistic_confounder_df0 = []
    auprc_logistic_confounder_df1 = []
    precision_confounder = []
    recall_confounder = []
    f1_confounder = []
    precision_confounder_df0 = []
    recall_confounder_df0 = []
    f1_confounder_df0 = []
    precision_confounder_df1 = []
    recall_confounder_df1 = []
    f1_confounder_df1 = []
    
    auprc_logistic_vanilla = []
    auprc_logistic_vanilla_df0 = []
    auprc_logistic_vanilla_df1 = []
    precision_vanilla = []
    recall_vanilla = []
    f1_vanilla = []
    precision_vanilla_df0 = []
    recall_vanilla_df0 = []
    f1_vanilla_df0 = []
    precision_vanilla_df1 = []
    recall_vanilla_df1 = []
    f1_vanilla_df1 = []
    
    valid_n_full_settings = []


    for iRun in range(runs):



        _rand = random.randint(0, 2**32 - 1)
        print(_rand)

        print(iRun)
        
        for c in tqdm(valid_full_settings, desc="Overall progress"):
        # for c in test_settings:


            c = c.copy()
            # create train/test split according to stats
            # dfs = create_mix(df0=df_wls_merge, df1=df_adress, target='label', setting= c, sample=False)
            # dfs = create_mix(df0=df_shac_uw, df1=df_shac_mimic, target=label, setting= c, sample=False, seed=random.randint(0,1000))
            dfs = create_mix(df0=df0, df1=df1, target=label, setting=c, sample=False, 
                             # seed=random.randint(0,1000),
                             seed=_rand
                            )

            if dfs is None:
                continue
            c['run'] = iRun
            

            
            y_train = dfs['train'][label]
            y_test = dfs['test'][label]

            n_test = len(y_test)

            df_test = dfs['test']


            #####################  Simple Logistic Regression, WITHOUT confounder
            tokenized_test = datasets_loader(dfs['test'])
            tokenized_test = tokenized_test.remove_columns(['text', 'dfSource', 'label'])
            
            y_probs_vanilla = []
            
            model.eval()

            test_dataloader = DataLoader(tokenized_test, batch_size=16, shuffle=False)
            
            for batch in tqdm(test_dataloader, desc="One Eval Cycle"):
                with torch.no_grad():
                    outputs = model(**batch)
                    
                    tmp = softmax(outputs['logits'].detach().cpu().type(torch.float)).numpy()

                    y_probs_vanilla.append(tmp)
            
            y_probs_vanilla = np.concatenate(y_probs_vanilla).astype("float")

            try:
                auprc_logistic_vanilla.append(metrics.average_precision_score(y_true=y_test, y_score=y_probs_vanilla[:,1]))
                auprc_logistic_vanilla_df0.append(metrics.average_precision_score(y_true=y_test[df_test[domain_col] == z_Categories[0]], y_score=y_probs_vanilla[df_test[domain_col] == z_Categories[0],1]))
                auprc_logistic_vanilla_df1.append(metrics.average_precision_score(y_true=y_test[df_test[domain_col] == z_Categories[1]], y_score=y_probs_vanilla[df_test[domain_col] == z_Categories[1],1]))
                # t_vanilla = precision_recall_fscore_support(y_true=y_test, y_pred=y_probs_vanilla[:,1]>0.5, average="binary", pos_label=1)
                # t_van_df0 = precision_recall_fscore_support(y_true=y_test[df_test[domain_col] == z_Categories[0]], y_pred=y_probs_vanilla[df_test[domain_col] == z_Categories[0],1]>0.5, average="binary", pos_label=1)
                # t_van_df1 = precision_recall_fscore_support(y_true=y_test[df_test[domain_col] == z_Categories[1]], y_pred=y_probs_vanilla[df_test[domain_col] == z_Categories[1],1]>0.5, average="binary", pos_label=1)
                # precision_vanilla.append(t_vanilla[0])
                # recall_vanilla.append(t_vanilla[1])
                # f1_vanilla.append(t_vanilla[2])
                # precision_vanilla_df0.append(t_van_df0[0])
                # recall_vanilla_df0.append(t_van_df0[1])
                # f1_vanilla_df0.append(t_van_df0[2])
                # precision_vanilla_df1.append(t_van_df1[0])
                # recall_vanilla_df1.append(t_van_df1[1])
                # f1_vanilla_df1.append(t_van_df1[2])
                
                valid_n_full_settings.append(c)
            except:
                logging.warning("failed on setting: " + str(c))
                
           
    ############  Put Results in DataFrame, with extra information (a little redundant)

    # organize results in DataFrame
    df_eval = pd.DataFrame({
                            "auprc_logistic_vanilla": auprc_logistic_vanilla,
                            "auprc_logistic_vanilla_df0": auprc_logistic_vanilla_df0,
                            "auprc_logistic_vanilla_df1": auprc_logistic_vanilla_df1,
                            
                           })


    for k in valid_n_full_settings[0]['mix_param_dict'].keys():
        df_eval[k] = [_dict['mix_param_dict'][k] for _dict in valid_n_full_settings]

    for k in valid_n_full_settings[0].keys():
        if k != "mix_param_dict":
            df_eval[k] = [_dict[k] for _dict in valid_n_full_settings]

    
    # outname = f"{outdir}/{transform}_{p_pos_train_z0}_{p_pos_train_z1}_{n_test}_{penalty}_C{C}_V{v}.pkl"
    outname = f"{outdir}/{transform}_ntest_{n_test}_{penalty}_C{C}_V{v}.pkl"

    with open(outname, "wb") as f:
        pickle.dump(df_eval, file=f)
        
