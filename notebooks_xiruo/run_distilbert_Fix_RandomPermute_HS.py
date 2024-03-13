import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import sys

sys.path.append("../src")

import pickle
import argparse
import pandas as pd
import numpy as np
import pathlib
import random
import itertools
from sklearn import metrics
from tqdm.auto import tqdm
from copy import deepcopy

import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer

import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertPreTrainedModel, PretrainedConfig, DistilBertModel
from transformers import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import SequenceClassifierOutput
from transformers import DistilBertTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModel

from typing import Optional
from torch.nn import MSELoss, CrossEntropyLoss
from typing import Union
from typing import Tuple
from scipy.special import softmax

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from utils import number_split, create_mix
from data_process import load_wls_adress_AddDomain
from process_SHAC import load_process_SHAC
from custom_distance import KL

### Temporary Argparse
parser = argparse.ArgumentParser()

# Adding optional argument
# parser.add_argument("-c", "--CombinationIdx", type=int, help="Set idx of c to use")
# parser.add_argument("-q", "--quantization", action="store_true")
# parser.add_argument("--lora_r", type=int, default=8, help="Set LoRA r value")
# parser.add_argument(
#     "--model_size", type=int, default=7, help="Llama 2 size: 7, 13, or 70"
# )

# Read arguments from command line
# args = parser.parse_args()


def permutePercent(in_list, fixed_pct=0.1, fixed_pos=None, seed=123):
    # fixed_pos: list of fixed positions
    N = len(in_list)

    if (fixed_pos is None) and fixed_pct:
        fixed_n = int(np.ceil(N * fixed_pct))
        random.seed(seed)
        fixed_pos = random.sample(range(0, N), fixed_n)

    elif (fixed_pos is None) and (fixed_pct is None):
        os.error("No Fixed Position Provided")

    ret_ls = [idx if idx in fixed_pos else x for idx, x in enumerate(in_list)]

    set_diff = set(in_list) - set(ret_ls)
    set_dups = [i for i in set(ret_ls) if ret_ls.count(i) > 1]

    while len(set_dups):
        a = set_dups.pop()
        r = set_diff.pop()
        for idx, x in enumerate(ret_ls):
            if (x == a) and (idx not in fixed_pos):
                ret_ls[idx] = r
                break
    return ret_ls


class PermDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_sources = 2
        self.config = config
        self.fixed_pct = config.fixed_pct
        self.permseed = config.permseed
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)

        # pemb[2] (in general, pemb[last]) is the identity permutation

        ret_ls = []
        for i in range(self.num_sources):
            while True:
                tmp_ts = torch.randperm(config.dim)
                tmp_ls = permutePercent(
                    in_list=tmp_ts.tolist(),
                    fixed_pct=self.fixed_pct,
                    seed=self.permseed,
                )
                if (tmp_ls not in ret_ls) and (
                    tmp_ls not in ([np.arange(self.num_sources).tolist()])
                ):
                    ret_ls.append(tmp_ls)
                    break
        self.pemb = torch.cat(
            (torch.tensor(ret_ls), torch.arange(0, config.dim).unsqueeze(0))
        )

        self.pemb = torch.nn.Parameter(self.pemb, requires_grad=False)

        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    # @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        source: Optional[int] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        if source is None:
            # Use the last
            perm_ids = self.num_sources * np.ones(pooled_output.shape[0])
        else:
            perm_ids = np.asarray(
                [source.cpu().numpy()]
            )  # for i in range(pooled_output.shape[0])])

        perms = self.pemb[perm_ids]
        sub_permuted = torch.gather(pooled_output, -1, perms)

        logits = self.classifier(sub_permuted)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # NOTE: to verify here
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = softmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    # compute AUPRC, based on only two levels of Y
    # predictions, labels = eval_pred
    # probabilities = nn.functional.softmax(torch.FloatTensor(predictions), dim=-1)[:,1]

    # auprc = average_precision_score(y_true=labels, y_score=probabilities)
    auprc = average_precision_score(y_true=labels, y_score=probs[:, 1])

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auprc": auprc,
    }


######  Load Data
### Hate Speech
z_category = ["nothate", "hate"]

y_Categories = [0, 1]
n_yCats = len(y_Categories)

label2id = {z: idx for idx, z in zip(range(len(z_category)), z_category)}
id2label = {idx: z for idx, z in zip(range(len(z_category)), z_category)}

# (1) dynGen
df_dynGen = pd.read_csv(
    "/bime-munin/xiruod/data/hateSpeech_Bulla2023/Dynamically-Generated-Hate-Speech-Dataset/Dynamically Generated Hate Dataset v0.2.3.csv",
)

df_dynGen["label"] = df_dynGen["label"].map({"hate": "hate", "nothate": "nothate"})
df_dynGen["dfSource"] = "dynGen"
df_dynGen["label_binary"] = df_dynGen["label"].map({"hate": 1, "nothate": 0})

# (2)  wsf
ls_allFiles = pathlib.Path(
    "/bime-munin/xiruod/data/hateSpeech_Bulla2023/hate-speech-dataset/all_files/"
).glob("*.txt")

ls_id = []
ls_text = []

for ifile in ls_allFiles:
    ls_id.append(ifile.name.split(".txt")[0])
    with open(ifile, "r") as f:
        ls_text.append(f.read())

df_wsf_raw = pd.DataFrame({"file_id": ls_id, "text": ls_text})

df_wsf_annotation = pd.read_csv(
    "/bime-munin/xiruod/data/hateSpeech_Bulla2023/hate-speech-dataset/annotations_metadata.csv"
)

df_wsf = df_wsf_raw.merge(df_wsf_annotation, on="file_id", how="inner")

df_wsf = df_wsf[df_wsf["label"].isin(["hate", "noHate"])].reset_index(drop=True)

df_wsf["label"] = df_wsf["label"].map({"hate": "hate", "noHate": "nothate"})
df_wsf["dfSource"] = "wsf"
df_wsf["label_binary"] = df_wsf["label"].map({"hate": 1, "nothate": 0})

##### Split
# Hate Speech Detection: df_dynGen (0.55) vs df_wsf (0.11)
n_test = 1000
train_test_ratio = 4


p_pos_train_z0_ls = [
    0.2,
    0.5,
    0.6,
    0.8,
    0.9,
]  # probability of training set examples drawn from site/domain z0 being positive
p_pos_train_z1_ls = [
    0.2,
    0.4,
    0.5,
    0.6,
    0.8,
    0.9,
]  # probability of test set examples drawn from site/domain z1 being positive

p_mix_z1_ls = [0.2, 0.4, 0.6, 0.8]  # = np.arange(0.1, 0.9, 0.05)

# alpha_test_ls = np.arange(0, 10, 0.05)

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
        # if not (round(number_setting['mix_param_dict']['C_y'], 4) in [0.36, 0.44, 0.52]):
        #     continue
        if np.all([number_setting[k] >= 10 for k in list(number_setting.keys())[:-1]]):
            valid_full_settings.append(number_setting)


class train_config:
    def __init__(self):
        self.quantization: bool = False


##### Run Experiments

import warnings

warnings.simplefilter("ignore")

globalconfig = train_config()
globalconfig.device = "cuda:0"
globalconfig.per_device_train_batch_size = 16
globalconfig.per_device_eval_batch_size = 16
globalconfig.logging_training = "../logsTensorBoard"
globalconfig.tokenizer_max_len = 512

runs = 5
num_train_epochs = 3
model_name = "DistilBERT"

### Hate Speech
z_Categories = ["dynGen", "wsf"]  # the order here matters! Should match with df0, df1
label = "label_binary"
n_zCats = len(z_Categories)
txt_col = "text"
domain_col = "dfSource"
df0 = df_dynGen
df1 = df_wsf
outdir = f"../output/DistilBERT_HateSpeechBalanceAlpha_10Pct_Fix_RandomPermute_ShorterVersion_1_20"
log_f = "../log/DistilBERT_HateSpeechBalanceAlpha_10Pct_Fix_RandomPermute_ShorterVersion_1_20.log"
globalconfig.fixed_pct = 0.1
globalconfig.permseed = 19

# NTOE: for shorter version!!!
valid_full_settings = [
    valid_full_settings[x] for x in list(range(len(valid_full_settings))) if x % 20 == 0
]

### SHAC
# z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
# label='Drug'
# n_zCats = len(z_Categories)
# txt_col="text"
# domain_col = "location"
# df0 = df_shac_uw
# df1 = df_shac_mimic
# outdir = f"../output/regressionSHACBalanceAlpha_SimplePermute_TestAverageByProportion"
# log_f = "../log/regressionSHACBalanceAlpha_SimplePermute_TestAverageByProportion.log"

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


##### Test for LLaMa Average Embeddings
## SHAC

# # transform = "LLaMaAverageV2_7B"
# transform = "LLaMaAverageV2_13B"
# transform = "LLaMaAverageV2_70B_8Quant"

# runs = 5

# # SHAC
# z_Categories = ["uw", "mimic"]  # the order here matters! Should match with df0, df1
# label='Drug'
# n_zCats = len(z_Categories)
# txt_col="LLaMaEmbeddings"
# domain_col = "location"
# df0 = df_shac_llama_average_uw
# df1 = df_shac_llama_average_mimic
# outdir = f"../output/regressionSHACBalanceAlpha"

## Hate Speech

# transform = "LLaMaAverageV2_7B"
# transform = "LLaMaAverageV2_13B"
# transform = "LLaMaAverageV2_70B_8Quant"
# transform = "LLaMaAverageV2_7B_Permute"
# transform = "LLaMaAverageV2_13B_Permute"


# runs = 5

# z_Categories = ["dynGen", "wsf"]  # the order here matters! Should match with df0, df1
# label='label_binary'
# n_zCats = len(z_Categories)
# txt_col="LLaMaEmbeddings"
# domain_col = "dfSource"
# df0 = df_dynGen
# df1 = df_wsf
# outdir = f"../output/regressionHateSpeechBalanceAlpha"


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df0["labels"] = le.fit_transform(df0[label])
df1["labels"] = le.fit_transform(df1[label])

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.max_len = globalconfig.tokenizer_max_len


# add labels and source to data for BERT
for _ in [df0, df1]:
    _["tokenized"] = _[txt_col].apply(
        lambda x: tokenizer(
            x,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=globalconfig.tokenizer_max_len,
        )
    )
    for i, row in _.iterrows():
        row["tokenized"].update(
            {"labels": torch.tensor(int(row["labels"]), dtype=torch.long)}
        )
        row["tokenized"].update(
            {
                "source": torch.tensor(
                    int(z_Categories.index(row[domain_col])), dtype=torch.long
                )
            }
        )  # NOTE: this only works for binary Category cases!!!
        row["tokenized"].update({"input_ids": row["tokenized"]["input_ids"].squeeze()})
        row["tokenized"].update(
            {"attention_mask": row["tokenized"]["attention_mask"].squeeze()}
        )


os.makedirs(outdir, exist_ok=True)


# setting for logistic regression
# penalty = "l1"
# solver = "liblinear"
# penalty = "l2"
# solver = "lbfgs"


random.seed(123)
auprc_logistic_confounder = []
auprc_logistic_confounder_df0 = []
auprc_logistic_confounder_df1 = []

auprc_logistic_vanilla = []
auprc_logistic_vanilla_df0 = []
auprc_logistic_vanilla_df1 = []

auprc_logistic_known_source = []
auprc_logistic_noPermute = []


valid_n_full_settings = []


precision_confounder = []
recall_confounder = []
f1_confounder = []
precision_confounder_df0 = []
recall_confounder_df0 = []
f1_confounder_df0 = []
precision_confounder_df1 = []
recall_confounder_df1 = []
f1_confounder_df1 = []

# precision_vanilla = []
# recall_vanilla = []
# f1_vanilla = []
# precision_vanilla_df0 = []
# recall_vanilla_df0 = []
# f1_vanilla_df0 = []
# precision_vanilla_df1 = []
# recall_vanilla_df1 = []
# f1_vanilla_df1 = []


config = DistilBertConfig(
    vocab_size=30522,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    hidden_size=768,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    pad_token_id=0,
    eos_token_id=2,
    bos_token_id=1,
    sep_token_id=3,
    cls_token_id=4,
    num_labels=2,
    problem_type="single_label_classification",
    output_attentions=False,
    output_hidden_states=False,
    use_cache=True,
    fixed_pct=globalconfig.fixed_pct,
    permseed=globalconfig.permseed,
)


for iRun in range(runs):
    _rand = random.randint(0, 2**32 - 1)
    print(_rand)

    print(iRun)
    for c in tqdm(valid_full_settings, file=open(log_f, "w")):
        # for c in test_settings:

        c = c.copy()

        # create train/test split according to stats
        # dfs = create_mix(df0=df_wls_merge, df1=df_adress, target='label', setting= c, sample=False)
        # dfs = create_mix(df0=df_shac_uw, df1=df_shac_mimic, target=label, setting= c, sample=False, seed=random.randint(0,1000))
        dfs = create_mix(
            df0=df0,
            df1=df1,
            target=label,
            setting=c,
            sample=False,
            # seed=random.randint(0,1000),
            seed=_rand,
        )

        if dfs is None:
            continue

        # #### TO DELETE: For results on Selected C_y ONLY!!!!!!!!!
        # if round(c['mix_param_dict']['C_y'], 4) not in [0.36, 0.44, 0.52, 0.24, 0.54, 0.84]:
        #     continue

        c["run"] = iRun
        valid_n_full_settings.append(c)

        y_train = dfs["train"][label]
        y_test = dfs["test"][label]

        n_test = len(y_test)

        df_test = dfs["test"].copy(deep=True)

        model = PermDistilBertForSequenceClassification(config)

        training_args = TrainingArguments(
            output_dir=outdir,  # output directory
            num_train_epochs=num_train_epochs,  # total number of training epochs
            per_device_train_batch_size=globalconfig.per_device_train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=globalconfig.per_device_eval_batch_size,  # batch size for evaluation
            warmup_steps=50,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir=globalconfig.logging_training,
            logging_strategy="steps",
            logging_steps=10,
            # evaluation_strategy="epoch",
            save_strategy="no",  # "epoch",
            learning_rate=5e-5,
            # eval_steps=10,
            # save_steps=10,
            # save_total_limit=2,
            # load_best_model_at_end=False,
            # metric_for_best_model='loss', # "f1"
            # greater_is_better = False,
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=dfs["train"]["tokenized"],  # training dataset
            eval_dataset=None,  # evaluation dataset
            compute_metrics=compute_metrics,
        )

        # Start training
        ret_train = trainer.train()
        # ret_eval = trainer.evaluate()

        # Predict

        ## SR0, SR1, SR2, ...

        for i in range(n_zCats + 1):
            df_test[f"tokenized_sr{i}"] = df_test.apply(
                lambda x: deepcopy(x["tokenized"]), axis=1
            )
            for _, row in df_test.iterrows():
                row[f"tokenized_sr{i}"]["source"] = torch.tensor(i, dtype=torch.long)

        # ret_pred = trainer.predict(test_dataset=df_test["tokenized"])
        # y_probs_confound = softmax(ret_pred.predictions, axis=1)

        y_probs_ls = []

        for i in range(n_zCats):
            _y_probs = trainer.predict(test_dataset=df_test[f"tokenized_sr{i}"])
            _y_probs = softmax(_y_probs.predictions, axis=1)
            y_probs_ls.append(_y_probs)

        # calculate P(Z): NOTE: this may not be useful, because it is pre-defined!!!!
        p_z = []

        for i in z_Categories:
            p_z.append(sum(dfs["train"][domain_col] == i) / len(dfs["train"]))

        # calculate P(Y|X): sum(P(y|x,z) * P(z))
        y_probs_confound = np.empty((n_test, n_yCats))
        y_probs_confound.fill(0)

        for i in range(n_zCats):
            y_probs_confound += y_probs_ls[i] * p_z[i]

        # save metrics
        ret = c
        ret.update(ret_train.metrics)
        trainer.save_metrics(split="all", metrics=ret)

        ret_code = 1

        # Predict No Permutation - use original embedding - no permutation
        y_probs_noPermute = trainer.predict(
            test_dataset=df_test[f"tokenized_sr{n_zCats}"]
        )
        y_probs_noPermute = softmax(y_probs_noPermute.predictions, axis=1)

        # Predict assume known Source
        y_probs_known_source = trainer.predict(test_dataset=df_test[f"tokenized"])
        y_probs_known_source = softmax(y_probs_known_source.predictions, axis=1)

        auprc_logistic_known_source.append(
            metrics.average_precision_score(
                y_true=y_test, y_score=y_probs_known_source[:, 1]
            )
        )
        auprc_logistic_noPermute.append(
            metrics.average_precision_score(
                y_true=y_test, y_score=y_probs_noPermute[:, 1]
            )
        )
        # auprc_logistic_vanilla.append(
        #     metrics.average_precision_score(
        #         y_true=y_test, y_score=y_probs_vanilla[:, 1]
        #     )
        # )
        auprc_logistic_confounder.append(
            metrics.average_precision_score(
                y_true=y_test, y_score=y_probs_confound[:, 1]
            )
        )
        auprc_logistic_confounder_df0.append(
            metrics.average_precision_score(
                y_true=y_test[df_test[domain_col] == z_Categories[0]],
                y_score=y_probs_confound[df_test[domain_col] == z_Categories[0], 1],
            )
        )
        auprc_logistic_confounder_df1.append(
            metrics.average_precision_score(
                y_true=y_test[df_test[domain_col] == z_Categories[1]],
                y_score=y_probs_confound[df_test[domain_col] == z_Categories[1], 1],
            )
        )
        t_confounder = precision_recall_fscore_support(
            y_true=y_test,
            y_pred=y_probs_confound[:, 1] > 0.5,
            average="binary",
            pos_label=1,
        )
        t_con_df0 = precision_recall_fscore_support(
            y_true=y_test[df_test[domain_col] == z_Categories[0]],
            y_pred=y_probs_confound[df_test[domain_col] == z_Categories[0], 1] > 0.5,
            average="binary",
            pos_label=1,
        )
        t_con_df1 = precision_recall_fscore_support(
            y_true=y_test[df_test[domain_col] == z_Categories[1]],
            y_pred=y_probs_confound[df_test[domain_col] == z_Categories[1], 1] > 0.5,
            average="binary",
            pos_label=1,
        )
        precision_confounder.append(t_confounder[0])
        recall_confounder.append(t_confounder[1])
        f1_confounder.append(t_confounder[2])
        precision_confounder_df0.append(t_con_df0[0])
        recall_confounder_df0.append(t_con_df0[1])
        f1_confounder_df0.append(t_con_df0[2])
        precision_confounder_df1.append(t_con_df1[0])
        recall_confounder_df1.append(t_con_df1[1])
        f1_confounder_df1.append(t_con_df1[2])


############  Put Results in DataFrame, with extra information (a little redundant)

# organize results in DataFrame
df_eval = pd.DataFrame(
    {
        "auprc_logistic_confounder": auprc_logistic_confounder,
        # "auprc_logistic_vanilla": auprc_logistic_vanilla,
        "auprc_logistic_confounder_df0": auprc_logistic_confounder_df0,
        "auprc_logistic_confounder_df1": auprc_logistic_confounder_df1,
        "precision_confounder": precision_confounder,
        "recall_confounder": recall_confounder,
        "f1_confounder": f1_confounder,
        "precision_confounder_df0": precision_confounder_df0,
        "recall_confounder_df0": recall_confounder_df0,
        "f1_confounder_df0": f1_confounder_df0,
        "precision_confounder_df1": precision_confounder_df1,
        "recall_confounder_df1": recall_confounder_df1,
        "f1_confounder_df1": f1_confounder_df1,
        "auprc_logistic_known_source": auprc_logistic_known_source,
        "auprc_logistic_noPermute": auprc_logistic_noPermute,
        # "auprc_logistic_vanilla_df0": auprc_logistic_vanilla_df0,
        # "auprc_logistic_vanilla_df1": auprc_logistic_vanilla_df1,
        # "precision_vanilla":precision_vanilla,
        # "recall_vanilla":recall_vanilla,
        # "f1_vanilla":f1_vanilla,
        # "precision_vanilla_df0":precision_vanilla_df0,
        # "recall_vanilla_df0":recall_vanilla_df0,
        # "f1_vanilla_df0":f1_vanilla_df0,
        # "precision_vanilla_df1":precision_vanilla_df1,
        # "recall_vanilla_df1":recall_vanilla_df1,
        # "f1_vanilla_df1":f1_vanilla_df1,
    }
)


for k in valid_n_full_settings[0]["mix_param_dict"].keys():
    df_eval[k] = [_dict["mix_param_dict"][k] for _dict in valid_n_full_settings]

for k in valid_n_full_settings[0].keys():
    if k != "mix_param_dict":
        df_eval[k] = [_dict[k] for _dict in valid_n_full_settings]


outname = f"{outdir}/{model_name}_ntest_{n_test}.pkl"

with open(outname, "wb") as f:
    pickle.dump(df_eval, file=f)
