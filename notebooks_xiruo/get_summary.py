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
import itertools
sys.path.append("../src")
from custom_distance import KL, conditionKL

warnings.filterwarnings("ignore")


def get_coefLog10(x, y):
    x = np.log10(x)
    x = sm.add_constant(x)

    lr_fit = sm.OLS(y, x)
    results = lr_fit.fit()

    return results.params[-1], results


# save

# with open("../output/regressionSHAC/05_02_400_L2_C5_V1.pkl", "wb") as f:
#     pickle.dump(df_eval, file=f)

# with open(f"../output/regressionInverseSHAC_MIMIC_UW/binaryUnigram_02_05_500_l2_C1_V100.pkl", "rb") as f:
# with open(f"../output/regressionSHAC/binaryUnigram_05_02_500_l2_C1_V1.pkl", "rb") as f:


##### SHAC

# fname="../output/regressionSHAC/binaryUnigram_0.3_0.3_500_l2_C1_V10"
# fname="../output/regressionSHAC/binaryUnigram_05_02_500_l2_C1_V10"
# fname="../output/regressionSHAC/Sentence-BERT_05_02_500_l2_C1_V10" ## AMIA Plot
# fname="../output/regressionSHAC/LLaMaAverage_0.5_0.2_500_l2_C1_V10"

# fname = "../output/regressionSHACBalanceAlpha/binaryUnigram_ntest_200_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/Sentence-BERT_ntest_200_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_12720_ntest_200_l2_C1_V10" ## Balanced!
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_l2_C1_V10" ## Balanced!
# fname = "../output/regressionSHACBalanceAlpha/LLaMaAverage_ntest_200_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/DistilBERT_12720_ntest_200_l2_C1_V10"

# fname="../output/regressionSHAC/Clinical-BERT_0.5_0.2_500_l2_C1_V10"

# fname = "../output/regressionSHACBalanceAlpha/LLaMaAverageV2_7B_ntest_200_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LLaMaAverageV2_13B_ntest_200_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LLaMaAverageV2_70B_8Quant_ntest_200_l2_C1_V10"


# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_7B-loraR-2_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_7B-loraR-8_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_7B-loraR-32_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_13B-loraR-2_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_13B-loraR-8_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_13B-loraR-32_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_13B-loraR-64_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_70B-loraR-2_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_70B-loraR-8_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_70B-loraR-32_l2_C1_V10"

# fname = "../output/regressionSHACBalanceAlpha/Sentence-BERT_0.5_0.2_500_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha/Sentence-BERT_ntest_500_l2_C1_V10"

# fname = "../output/regressionSHACBalanceAlpha_SimplePermuteAverage/Sentence-BERT_ntest_500_l2_C1_V10"
# fname = "../output/regressionSHACBalanceAlpha_SimplePermute_TestAverageByProportion/Sentence-BERT_ntest_500_l2_C1_V10"

# fname = "../output/DistilBERT_SHACBalanceAlpha_RandomPermute_ShorterVersion_1_20/DistilBERT_ntest_500"
# fname = "../output/DistilBERT_SHACBalanceAlpha_10Pct_Fix_RandomPermute_ShorterVersion_1_5/DistilBERT_ntest_500"


# fname = "../output/tmpData/LoraPredict/set-1355-quantization-epoch3-llama-2-7B-loraR-8_ntest_500_setting_1_15"
# fname = "../output/tmpData/LoraPredict_Original_Target/set-1355-quantization-epoch3-llama-2-7B-loraR-8_ntest_500_setting_1_15"
# fname = "../output/tmpData/LoraAdapters_TargetNorm/set-1355-quantization-epoch3-llama-2-7B-loraR-8_ntest_500_setting_1_15"
# fname = "../output/tmpData/LoraPredict/set-1355-quantization-epoch3-llama-2-13B-loraR-8_ntest_500_setting_1_15"
# fname = "../output/tmpData/LoraPredict_Original_Target/set-1355-quantization-epoch3-llama-2-13B-loraR-8_ntest_500_setting_1_15"
# fname = "../output/tmpData/LoraAdapters_TargetNorm/set-1355-quantization-epoch3-llama-2-13B-loraR-8_ntest_500_setting_1_15"
# fname = "../output/tmpData/LoraAdapters_TargetFroNorm/set-1355-quantization-epoch3-llama-2-7B-loraR-8_ntest_500_setting_1_15"
# fname = "../output/tmpData/LoraAdapters_TargetFroNorm/set-1355-quantization-epoch3-llama-2-13B-loraR-8_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_1_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1-added_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_1_Added/set-1355-quantization-epoch3-llama-2-13B-loraR-8-gamma_1-added_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_0.1_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_0_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_0.2_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_0_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_0.5_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_0_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_0.8_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_0_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_1.5_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_2.0_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_2_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_3.0_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_3_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_1.0_Lamda1_1.0_Lambda2_0.5_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1.0-lambda1_1.0-lambda2_0.5-added_ntest_500_setting_1_15"
# fname = "../output/tmpData/WeightsEdited_Gamma_1.0_Lamda1_2.0_Lambda2_1.0_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1.0-lambda1_2.0-lambda2_1.0-added_ntest_500_setting_1_20"
# fname = "../output/tmpData/WeightsEdited_Gamma_1.0_Lamda1_1.0_Lambda2_0.0_Added/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1.0-lambda1_1.0-lambda2_0.0-added_ntest_500_setting_1_20"

# fname = "../output/tmpData/SHAC/OriginalWeightsEdited-set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_500-pct_1_20"
# fname = "../output/tmpData/SHAC/OriginalWeightsEdited-set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_500-pct_1_20"
# fname = "../output/tmpData/SHAC/OriginalWeightsEdited-set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added-ntest_500-pct_1_15"

# fname = "../output/tmpData/SHAC/OriginalWeightsEdited-set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_200-pct_1_5"
# fname = "../output/tmpData/SHAC/OriginalWeightsEdited-set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added-ntest_200-pct_1_5"
# fname = "../output/tmpData/SHAC/OriginalWeightsEdited-set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_200-pct_1_5"

# fname = "../output/tmpData/SHAC/OriginalWeightsEdited-set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_200-pct_1_1_wrongSet"
# fname = "../output/tmpData/SHAC/OriginalWeightsEdited-set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added-ntest_200-pct_1_1_wrongSet"

# fname = '../output/regression_onlyAlpha_Train_1_SHACBalanceAlpha/binaryUnigram_ntest_200_l2_C1_V10'
# fname = '../output/regressionStratifiedSHACBalanceAlpha/binaryUnigram_ntest_200_l2_C1_V10'
# fname = '../output/regressionOversamplingSHACBalanceAlpha/binaryUnigram_ntest_200_l2_C1_V10'
# fname = '../output/regression_Sampling_0.8_SHACBalanceAlpha/Multiplier_0.8-binaryUnigram-ntest_200-l2-C1-V10'
# fname = '../output/regression_ReSampling_SHACBalanceAlpha/Multiplier_1-p_pos_train_z0_target_0.2-p_pos_train_z1_target_0.2-binaryUnigram-ntest_200-l2-C1-V10'
# fname = '../output/regression_ReSampling_SHACBalanceAlpha/Multiplier_1-p_pos_train_z0_target_0.4-p_pos_train_z1_target_0.4-binaryUnigram-ntest_200-l2-C1-V10'
# fname = '../output/regression_ReSampling_SHACBalanceAlpha/Multiplier_1-p_pos_train_z0_target_0.4-p_pos_train_z1_target_0.4-Sentence-BERT-ntest_200-l2-C1-V10'
# fname = '../output/regression_ReSampling_SHACBalanceAlpha/Multiplier_1-p_pos_train_z0_target_0.6-p_pos_train_z1_target_0.6-binaryUnigram-ntest_200-l2-C1-V10'


# fnames = [
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_200-Runs_5",
# ]

# fnames = [
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added-ntest_200-Runs_5",

# ]
# subdir = "original"



# fnames = [
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added-ntest_200-Runs_5",

#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_200-Runs_5",
# ]
# fnames = [
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_ReverseSource/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added-ntest_200-Runs_5",
    
# ]
# subdir = "ReverseSource"

# fnames = [
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-lambda3_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-lambda3_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-lambda3_1.0-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.5-lambda3_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.5-lambda3_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.5-lambda3_0.5-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.6-lambda2_0.3-lambda3_0.3-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.6-lambda2_0.3-lambda3_0.3-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.6-lambda2_0.3-lambda3_0.3-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.6-lambda2_0.8-lambda3_0.8-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.6-lambda2_0.8-lambda3_0.8-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.6-lambda2_0.8-lambda3_0.8-added-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added-ntest_200-Runs_5",
#     "../output/tmpData/SHAC/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added-ntest_200-Runs_5",
    

# ]
# subdir = "threeLambdas"

# fnames = [
#     "../output/tmpData/SHAC_proj/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_S-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_S-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_RS-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_RS-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC_proj/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_S-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_S-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_RS-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_RS-ntest_200-Runs_5",
    
   
    

# ]
# subdir = "threeLambdas_proj"

# fnames = [
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_7B-loraR-2_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_7B-loraR-8_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_7B-loraR-32_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_13B-loraR-2_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_13B-loraR-8_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_13B-loraR-32_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_13B-loraR-64_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_70B-loraR-2_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_70B-loraR-8_l2_C1_V10",
#     "../output/regressionSHACBalanceAlpha/LoRA_18965_ntest_200_70B-loraR-32_l2_C1_V10",
# ]
# subdir = "loras"

# fnames = [
    
#     "../output/tmpData/SHAC_norm_proj/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_RS-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_norm_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_RS-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_norm_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_S-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_norm_proj/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_S-ntest_200-Runs_5",

# ]
# subdir = "threeLambdas_norm_proj"

# fnames = [
#     "../output/tmpData/SHAC_norm/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_norm/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_norm/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC_norm/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_norm/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_norm/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-ntest_200-Runs_5",

# ]
# subdir = "threeLambdas_norm"

# fnames = [
#     "../output/tmpData/SHAC_proj/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_RS-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_RS-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_1.0-lambda3_0.3-added-Proj_S-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_1.0-lambda3_0.3-added-Proj_S-ntest_200-Runs_5",
    
#     "../output/tmpData/SHAC_proj/Eval-set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_RS-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_RS-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-lambda3_0.5-added-Proj_S-ntest_200-Runs_5",
#     "../output/tmpData/SHAC_proj/Eval-set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-lambda3_0.5-added-Proj_S-ntest_200-Runs_5",
    
# ]
# subdir = "threeLambdas_proj"




# dataset_name = "SHAC"


##############  HateSpeech
# fname = "../output/regressionHateSpeech/Sentence-BERT_0.6_0.1_500_l2_C1_V10"

# fname = "../output/regressionHateSpeechBalanceAlpha/Sentence-BERT_0.6_0.1_500_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha/Sentence-BERT_ntest_5000_l2_C1_V10"

# fname = "../output/regressionHateSpeechBalanceAlpha/binaryUnigram_ntest_1000_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha/Sentence-BERT_ntest_1000_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha/LoRA_1244_ntest_1000_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha/LoRA_2234_ntest_1000_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha/LoRA_75_ntest_1000_l2_C1_V10" ## Balanced
# fname = "../output/regressionHateSpeechBalanceAlpha/DistilBERT_75_ntest_1000_l2_C1_V10" ## Balanced
# fname = "../output/regressionHateSpeechBalanceAlpha/LLaMaAverageV2_7B_ntest_1000_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha/LLaMaAverageV2_13B_ntest_1000_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha/LLaMaAverageV2_70B_8Quant_ntest_1000_l2_C1_V10"

# fname = "../output/regressionHateSpeechBalanceAlpha/LLaMaAverageV2_7B_Permute_ntest_1000_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha_SimplePermute/Sentence-BERT_ntest_1000_l2_C1_V10"

# fname = "../output/regressionHateSpeechBalanceAlpha_SimplePermuteAverage/Sentence-BERT_ntest_1000_l2_C1_V10"
# fname = "../output/regressionHateSpeechBalanceAlpha_SimplePermute_TestAverageByProportion/Sentence-BERT_ntest_1000_l2_C1_V10"

# fname = "../output/DistilBERT_HateSpeechBalanceAlpha_RandomPermute_ShorterVersion_1_5/DistilBERT_ntest_1000"
# fname = "../output/DistilBERT_HateSpeechBalanceAlpha_RandomPermute_ShorterVersion_1_20/DistilBERT_ntest_1000"
# fname = "../output/DistilBERT_HateSpeechBalanceAlpha_10Pct_Fix_RandomPermute_ShorterVersion_1_20/DistilBERT_ntest_1000"

# fname = '../output/regressionStratifiedHateSpeechBalanceAlpha/binaryUnigram_ntest_1000_l2_C1_V10'

# fname = '../output/regressionOversamplingHateSpeechBalanceAlpha/binaryUnigram_ntest_1000_l2_C1_V10'
# fname = '../output/regression_Sampling_0.8_HateSpeechBalanceAlpha/Multiplier_0.8-binaryUnigram-ntest_1000-l2-C1-V10'
# fname = '../output/regression_ReSampling_HateSpeechBalanceAlpha/Multiplier_1-p_pos_train_z0_target_0.2-p_pos_train_z1_target_0.2-Sentence-BERT-ntest_1000-l2-C1-V10'
# fname = '../output/regression_ReSampling_HateSpeechBalanceAlpha/Multiplier_1-p_pos_train_z0_target_0.6-p_pos_train_z1_target_0.6-binaryUnigram-ntest_200-l2-C1-V10'

# fname = '../output/tmpData/HateSpeech/OriginalWeightsEdited-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_1000-pct_1_25'

# fnames = [
#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_1000-Runs_5',


#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added-ntest_1000-Runs_5',
# ]


# fnames = [
#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added-ntest_1000-Runs_5',

#     '../output/tmpData/HateSpeech/Eval-set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added-ntest_1000-Runs_5',
#     '../output/tmpData/HateSpeech/Eval-set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added-ntest_1000-Runs_5',


# ]

# fnames = [

#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added-ntest_200-Runs_5",
    
    
# ]
# subdir = "original"



# fnames = [
    
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech_ReverseSource/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added-ntest_200-Runs_5",
    
    
# ]
# subdir = "ReverseSource"


# fnames = [
    
#     "../output/tmpData/HateSpeech/Eval-set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added-ntest_200-Runs_5",
#     "../output/tmpData/HateSpeech/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added-ntest_200-Runs_5",
# ]


# subdir = "threeLambdas"

# fnames = [
#     "../output/tmpData/HateSpeech_norm_proj/Eval-set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_S-ntest_200-Runs_5",
    
# ]
# subdir = "threeLambdas_norm_proj"

# dataset_name = 'HateSpeech'



# dataset_name = 'SHAC'
# set_ls = [1152, 6114, 11063]

dataset_name = 'HateSpeech'
set_ls = [566, 3636, 6621]


lambda2_ls = ['0.0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0', '2.5', '3.0']

original_template = "../output/tmpData/{}/Eval-set-{}-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_{}-added-ntest_200-Runs_5"
subdir = "original"
# original_template = "../output/tmpData/{}_ReverseSource/Eval-set-{}-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_{}-added-ntest_200-Runs_5"
# subdir = "ReverseSource"

fnames = [original_template.format(dataset_name, s, l) for s, l in itertools.product(set_ls, lambda2_ls)]



def getSummary(fname, dataset_name):
    with open(f"{fname}.pkl", "rb") as f:
        df_eval = pickle.load(file=f)

    df = df_eval.copy()

    setname = int(fname.split("set-")[1].split("-")[0])
    if setname in [11063, 9870, 6621]:
        alpha_train = 0.2
    elif setname in [6114, 6126, 3636]:
        alpha_train = 1
    elif setname in [1152, 1874, 566]:
        alpha_train = 5

    # alpha_train = 1
    
    alpha_train_recip = 1 / alpha_train

    _df = df.copy()
    _df["C_y1"] = np.floor(_df["C_y"] * 10) / 10
    _df["C_y"] = _df["C_y"].round(2)

    dfgrp = _df.groupby(["C_y1", "C_y"], as_index=False).size()
    dfgrp = dfgrp.sort_values("size", ascending=False)
    dfgrp = dfgrp.groupby("C_y1", as_index=False).head(1)

    y_name = "auprc_weightsEdited"
    # y_name = "auprc_logistic_vanilla"

    x_name = "alpha_test"

    _df_plt = _df[_df["C_y"].isin(dfgrp["C_y"])].copy()

    cy_levels = _df_plt["C_y"].nunique()

    my_palette = sns.color_palette("crest", cy_levels)

    pred_ls = []
    pred_lower_ls = []
    pred_upper_ls = []
    pred_se_ls = []
    coef_lower_ls = []
    coef_upper_ls = []
    coef_ls = []
    cy_ls = []

    y_min_ls = []
    y_alpha_train = []
    y_alpha_train_recip = []

    for item, color in zip(_df_plt.groupby("C_y"), my_palette):
        for iy, ylabel in enumerate([y_name]):

            reg = get_coefLog10(x=list(item[1][x_name]), y=list(item[1][ylabel]))
            # pred = reg[1].predict([1,np.log10(np.quantile(sorted(item[1][x_name]),0.5))])[0]

            results = (
                reg[1]
                .get_prediction(
                    [1, np.log10(np.quantile(sorted(item[1][x_name]), 0.5))]
                )
                .summary_frame(alpha=0.05)
            )

            pred = results.iloc[0]["mean"]
            pred_lower = results.iloc[0]["mean_ci_lower"]
            pred_upper = results.iloc[0]["mean_ci_upper"]
            pred_se = results.iloc[0]["mean_se"]
            coef_lower = reg[1].conf_int()[1, :][0]
            coef_upper = reg[1].conf_int()[1, :][1]

            y_min = min(list(item[1][ylabel]))

            pred_lower_ls.append(pred_lower)
            pred_upper_ls.append(pred_upper)
            pred_ls.append(pred)
            pred_se_ls.append(pred_se)
            coef_lower_ls.append(coef_lower)
            coef_upper_ls.append(coef_upper)

            coef = reg[0]
            coef_ls.append(coef)
            cy_ls.append(item[0])

            y_min_ls.append(y_min)

            tmpdf = item[1]
            y_alpha_train.append(
                np.mean(tmpdf[tmpdf["alpha_test"] == alpha_train][y_name])
            )
            y_alpha_train_recip.append(
                np.mean(tmpdf[tmpdf["alpha_test"] == alpha_train_recip][y_name])
            )


    df_summary = pd.DataFrame(
        {
            "cy": cy_ls,
            "coef": coef_ls,
            "predMidpoint": pred_ls,
            "pred_lower": pred_lower_ls,
            "pred_upper": pred_upper_ls,
            "pred_se": pred_se_ls,
            "coef_lower": coef_lower_ls,
            "coef_upper": coef_upper_ls,
            "y_min": y_min_ls,
            "y_alpha_train": y_alpha_train,
            "y_alpha_train_recip": y_alpha_train_recip,
        }
    )
    name_base = Path(fname).name
    df_summary["name"] = name_base

    # df_summary['set'] = name_base.split("_")[1]
    # df_summary['model_size'] = [x for x in name_base.split("_") if 'loraR' in x][0].split("-")[0]
    # df_summary['loraR'] = int([x for x in name_base.split("_") if 'loraR' in x][0].split("-")[2])

    df_summary["set"] = name_base.split("-")[2]
    df_summary["lambda1"] = float(
        [x for x in name_base.split("-") if x.startswith("lambda1")][0].split("_")[1]
    )
    if 'lambda2' in fname:
        df_summary["lambda2"] = float(
            [x for x in name_base.split("-") if x.startswith("lambda2")][0].split("_")[1]
        )

    if "lambda3" in fname:
        df_summary["lambda3"] = float(
            [x for x in name_base.split("-") if x.startswith("lambda3")][0].split("_")[
                1
            ]
        )

    df_summary["pred_lowerdiff"] = df_summary["predMidpoint"] - df_summary["pred_lower"]
    df_summary["pred_upperdiff"] = df_summary["pred_upper"] - df_summary["predMidpoint"]

    df_summary["coef_lowerdiff"] = df_summary["coef"] - df_summary["coef_lower"]
    df_summary["coef_upperdiff"] = df_summary["coef_upper"] - df_summary["coef"]

    outdir = f"../output/tmpData/{dataset_name}/summary/{subdir}"
    os.makedirs(outdir, exist_ok=True)

    df_summary.to_csv(f"{outdir}/{name_base}.csv")


if __name__ == "__main__":
    for fname in fnames:
        getSummary(fname, dataset_name=dataset_name)
