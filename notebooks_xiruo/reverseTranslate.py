import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, choices=["SHAC", "HateSpeech", "CD"], help="dataset name"
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="On which GPU to run",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=512,
    help="max length for sentence",
)
args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu




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


# # dataset_name = "CD"
# dataset_name = "HateSpeech"
dataset_name = args.dataset
max_length = args.max_length

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

    
df0['ssid'] = ["df0_" + str(x) for x in np.arange(len(df0))]
df1['ssid'] = ["df1_" + str(x) for x in np.arange(len(df1))]


src = 'en'
tgt = 'de'

outdir = f"../output/ReverseTranslate/{dataset_name}"
os.makedirs(outdir, exist_ok=True)


tmp = reverseTranslationDF(df_in=df0, src=src, tgt=tgt, txt_col=txt_col, save=True, outfile=f"{outdir}/df0_{tgt}.csv", device="cuda:0", max_length=max_length)
tmp = reverseTranslationDF(df_in=df1, src=src, tgt=tgt, txt_col=txt_col, save=True, outfile=f"{outdir}/df1_{tgt}.csv", device="cuda:0", max_length=max_length)