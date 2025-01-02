import os
import argparse

parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument(
    "--dataset",
    type=str,
    default="CD",
    help="Dataset for the experiment",
)
parser.add_argument("-c", "--CombinationIdx", type=int, help="Set idx of c to use")
parser.add_argument("--batchSize", type=int, default=8, help="Batch size")
args = parser.parse_args()


from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import math
import scipy
from pathlib import Path
import sys
import os
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
from copy import deepcopy
sys.path.append("../src")
from custom_distance import KL, conditionKL
import itertools
import pickle

from utils import number_split, create_mix, appendMetrics
from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC
from custom_distance import KL
from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_CD import load_cd

import random
from tqdm import tqdm

warnings.filterwarnings("ignore")


from augmentation import reverseTranslationDF

import json
import requests

from tqdm import tqdm

sys.path.append("../src")
sys.path.append("../config")

from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT


dataset_name = args.dataset
# dataset_name = "HateSpeech"
# dataset_name = "SHAC"


batch_size = args.batchSize

CombinationIdx = args.CombinationIdx


######## Load Data
if dataset_name == "SHAC":
    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x["Drug"] else 0, axis=1)
    df_shac["dfSource"] = df_shac["location"]

    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

    n_test = 200
    
    z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
    label = "Drug"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "location"
    df0 = df_shac_uw
    df1 = df_shac_mimic
    
    df_split_label = "Drug"
    
    c = SHAC_DICT[f"c_n{n_test}_{CombinationIdx}"]
    
elif dataset_name == "HateSpeech":
    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()

    # n_test = 1000
    n_test = 200
    
    z_Categories = [
        "dynGen",
        "wsf",
    ]  # the order here matters! Should match with df0, df1
    label = "label_binary"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "dfSource"
    df0 = df_dynGen
    df1 = df_wsf
    
    df_split_label = "label_binary"
    
    c = HateSpeech_DICT[f"c_n{n_test}_{CombinationIdx}"]
    
elif dataset_name == "CD":
    df_all = load_cd()
    df_avh = df_all["avh"]
    df_r56 = df_all["r56"]

    n_test = 200
    
    z_Categories = ["avh", "r56"]
    label = "label_binary"
    n_zCats = len(z_Categories)
    txt_col = "text"
    domain_col = "dfSource"
    df0 = df_avh
    df1 = df_r56
    
    df_split_label = "label_binary"
    
    c = CD_DICT[f"c_n{n_test}_{CombinationIdx}"]

    
df0['ssid'] = ["df0_" + str(x) for x in np.arange(len(df0))]
df1['ssid'] = ["df1_" + str(x) for x in np.arange(len(df1))]





dfs = create_mix(
    df0=df0,
    df1=df1,
    target=df_split_label,
    setting=c,
    sample=False,
    # seed=random.randint(0,1000),
    seed=222,
)


outdir = f"../output/LLM_Generate/{dataset_name}"
os.makedirs(outdir, exist_ok=True)

url = "http://localhost:8077/"


instruction = ["You are a helpful assistant that rephrase text and make sentence smooth. I will give you a sample in the next paragraph, please give me 10 rephrased answers.\n\n"]

with requests.Session() as r:

    for tmpdf, df_name in zip([dfs['train']], [f'sub-n_{n_test}-set_{CombinationIdx}']):

        instruction_in = instruction * batch_size

        df_splits = np.array_split(range(len(tmpdf)), len(tmpdf) // batch_size)

        print(f"===============  Working on DF: {df_name} ===============\n")

        for i_split in tqdm(df_splits, file=open(f"../log/LLM_Generate_{dataset_name}_{df_name}.txt", "w")):
            inputx = ["Sample text: " + x for x in tmpdf.iloc[i_split][txt_col]]

            myobj = {'model': 'instruct',
                 'task': 'instruct',
                 'instruction': instruction_in,
                 'input': inputx,
                 'max_new_tokens' : 256,
                 'temperature' : 0.01,
                  'stop_conditions' : [128009],
                        }


            outputs = r.post(url, json=myobj, headers={"Connection": "close"})
            outputs = outputs.json()

            for i, x in zip(i_split, outputs):
                tmpdf.loc[tmpdf["ssid"] == tmpdf.iloc[i]["ssid"], "LLM_Generate"] = x

            # break


        tmpdf.to_csv(f"{outdir}/{df_name}.csv", index=False)

