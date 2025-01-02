# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="SHAC",
    help="Dataset for the experiment",
)
parser.add_argument("-c", "--CombinationIdx", type=int, help="Set idx of c to use")
parser.add_argument(
    "--num_train_epochs",
    type=int,
    default=10,
    help="Number of training epochs",
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="On which GPU to run",
)


args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# +

import sys


sys.path.append("../src")
sys.path.append("../config")

# +
from utils import number_split, create_mix, appendMetrics
import random
from copy import deepcopy
from tqdm.auto import tqdm

from process_CD import load_cd


import itertools

from transformers import (
    
    AutoTokenizer,
)

import torch
from torch.utils.data import Dataset, DataLoader

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

# -
import numpy as np
import pandas as pd

from process_CD import load_cd
from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_SHAC import load_process_SHAC
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT




dataset = args.dataset
CombinationIdx = args.CombinationIdx
num_train_epochs = args.num_train_epochs


# # Load Data and Split

# # +
# # dataset = "HateSpeech"
# dataset = "CD"
# CombinationIdx = 566
# # # CombinationIdx = 3636
# # CombinationIdx = 6621
# num_train_epochs = 20

# dataset = "SHAC"
# # CombinationIdx = 1152
# # CombinationIdx = 6114
# CombinationIdx = 11063
# num_train_epochs = 6

# -

n_test = 200
train_test_ratio = 4



# +
pick_C = CombinationIdx

if dataset == "SHAC":
    label = "Drug"
    
    domain_col = "location"


    df_shac = load_process_SHAC(replaceNA="all")
    df_shac["label_binary"] = df_shac.apply(lambda x: 1 if x[label] else 0, axis=1)
    df_shac["dfSource"] = df_shac[domain_col]
    
    
    df_shac_uw = df_shac.query("location == 'uw'").reset_index(drop=True)
    df_shac_mimic = df_shac.query("location == 'mimic'").reset_index(drop=True)

    df0 = df_shac_uw
    df1 = df_shac_mimic
    # df_split_label = "Drug"
    df_split_label = "label_binary"

    c = SHAC_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_2800"

    z_category = ["uw", "mimic"]
    y_Categories = [0, 1]
    txt_col = "text"
    
    
    
    
elif dataset == "HateSpeech":
    
    label = "label"
    ## Hate Speech data already have "label_binary" and dfSource
    df_dynGen = load_HateSpeech_dynGen()
    df_wsf = load_HateSpeech_wsf()
    
    
    df0 = df_dynGen
    df1 = df_wsf
    df_split_label = "label_binary"

    c = HateSpeech_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n1000_9870"

    z_category = ["dynGen", "wsf"]
    y_Categories = [0, 1]
    txt_col = "text"
    domain_col = "dfSource"
    
elif dataset == "CD":
    
    label = "label"

    df_all = load_cd()
    df_avh = df_all["avh"]
    df_r56 = df_all["r56"]
    
    
    df0 = df_avh
    df1 = df_r56
    df_split_label = "label_binary"

    c = CD_DICT[f"c_n{n_test}_{pick_C}"]  # e.g.: "c_n200_566"

    z_category = ["avh", "r56"]
    y_Categories = [0, 1]
    txt_col = "text"
    domain_col = "dfSource"






# +
# df_all = load_cd()
# df_avh = df_all["avh"]
# df_r56 = df_all["r56"]
# label = "label"
# z_category = ["avh", "r56"]
# y_Categories = [0, 1]

# df_split_label = "label_binary"
# txt_col = 'text'
# domain_col = "dfSource"

# df0 = df_avh
# df1 = df_r56



# +
# c = CD_DICT[f"c_n{200}_{566}"]

dfs = create_mix(
    df0=df0,
    df1=df1,
    target=df_split_label,
    setting=c,
    sample=False,
    # seed=random.randint(0,1000),
    seed=222,
)

# +


##### Experiment - For DistMatch

random.seed(12)
_rand = random.randint(0, 2**32 - 1)

df_training = deepcopy(dfs["train"])


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


# +




# +
## DistMatch
names_to_aug_ls = ["n_z0_pos_train", "n_z0_neg_train", "n_z1_pos_train", "n_z1_neg_train"]
df_ls = [df0_train_pos, df0_train_neg, df1_train_pos, df1_train_neg]


n_augs_dict = {}
for idx, (_, _df) in enumerate(zip(names_to_aug_ls, df_ls)):
    n_augs_dict[_] = c_train_target[_] - c[_]
    
    if n_augs_dict[_] < 0:
        df_ls[idx] = df_ls[idx].sample(
            n=c_train_target[_],
            random_state=_rand,
            replace=False,
            ignore_index=True,
        )

augMethod = "mixup"

# +
aug_query_dict = {
    
    "n_z0_pos_train": {"labels": True, "domain_idx": 0}, 
    "n_z0_neg_train": {"labels": False, "domain_idx": 0}, 
    "n_z1_pos_train": {"labels": True, "domain_idx": 1}, 
    "n_z1_neg_train": {"labels": False, "domain_idx": 1}, 
}


# -

n_augs_dict







from transformers import get_scheduler
from transformers import AdamW
from torch.nn import CrossEntropyLoss


# +

class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.model_name = 'roberta-base'
globalconfig.max_seq_length = 512
globalconfig.device = "cpu"
globalconfig.batch_size = 80#200
# globalconfig.device = "cuda"
# globalconfig.batch_size = 100

globalconfig.lr = 1e-4
globalconfig.weight_decay = 1e-3
globalconfig.num_train_epochs = num_train_epochs

globalconfig.momentum = 0.9


globalconfig.alpha = 4


# -



# +
globalconfig.output_dir = f"/bime-munin/xiruod/{globalconfig.model_name}_{dataset}-FullFT-{augMethod}/n{n_test}/set-{pick_C}-epoch{globalconfig.num_train_epochs}-mixupAlpha_{globalconfig.alpha}"


# -

globalconfig.output_dir


class TextDataset(Dataset):
    def __init__(self, df_in, txt_col, label_col, domain_col, tokenizer, max_length):
        self.primary_label = df_in[label_col]
        self.domain_label = df_in[domain_col].map({z_category[0]:0, z_category[1]:1})
        self.text = df_in[txt_col]
        self.tokenized = tokenizer(list(df_in[txt_col]), return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)

        self.tokenizer_keys = self.tokenized.keys()
        
    def __len__(self):
        return len(self.primary_label)
    
    def __getitem__(self, idx):
        primary_label = self.primary_label[idx]
        domain_label = self.domain_label[idx]
        text = self.text[idx]
        # input_ids = self.tokenized['input_ids'][idx]
        
        sample = {"Text": text, "labels": primary_label, "domain_label": domain_label}
        for key in self.tokenizer_keys:
            sample[key] = self.tokenized[key][idx]
        
        
        return sample

tokenizer = AutoTokenizer.from_pretrained(globalconfig.model_name, use_fast=False)


# +
dataset_train = TextDataset(df_in=dfs['train'], txt_col=txt_col, label_col=df_split_label, domain_col=domain_col, tokenizer=tokenizer, max_length=globalconfig.max_seq_length)

train_loader = DataLoader(dataset_train, batch_size=globalconfig.batch_size, shuffle=True)
# -



# # Model



from typing import List, Optional, Tuple, Union

import torch

from transformers import RobertaPreTrainedModel, RobertaModel

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, SequenceClassifierOutput


class RobertaForSequenceClassificationMixUp(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        self.n_aug_targets = {}
        self.n_aug_tracker = {}
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        domain_label: Optional[str] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = ()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        
        
        ## Add MixUp
        sequence_output_stack = []
        labels_mixup_stack = []
        for k,v in self.n_aug_targets.items():
            if (v > 0) and (self.n_aug_tracker[k] < v):
                
                indexMixUp = (labels == aug_query_dict[k]['labels'] ) & torch.tensor((np.array(domain_label.cpu()) == aug_query_dict[k]['domain_idx'])).to(globalconfig.device)
                
                sequence_output_forMixUp = sequence_output[indexMixUp]
                
                all_combos = list(itertools.combinations(range(sequence_output_forMixUp.shape[0]), 2))
                
                _ct = 0
                for _combos in all_combos:
                    
                    
                    _lambda = np.random.beta(globalconfig.alpha, globalconfig.alpha)
                    
                    sequence_output_stack.append( _lambda * sequence_output_forMixUp[_combos[0]] + (1-_lambda) * sequence_output_forMixUp[_combos[1]])
                    _ct += 1
                    
                    self.n_aug_tracker[k] += 1
                    
                    if self.n_aug_tracker[k] == v:
                        break
                        
                labels_mixup_stack.append([aug_query_dict[k]['labels']] * _ct)

        
        if labels is not None:
            if len(sequence_output_stack) > 0:
                sequence_output_mixup = torch.stack(sequence_output_stack)

                logits_mixup = self.classifier(sequence_output_mixup)
                

                labels_mixup = torch.tensor(np.hstack(labels_mixup_stack), dtype=torch.int64).to(globalconfig.device)

                loss_mixup = loss_fct(logits_mixup.view(-1, self.num_labels), labels_mixup.view(-1))

                loss_combine = loss + loss_mixup
                

            else:
                loss_combine = loss
        else:
            loss_combine = None
                    
        
        return SequenceClassifierOutput(
            loss=loss_combine,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    
    
    def inference(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = RobertaForSequenceClassificationMixUp.from_pretrained("roberta-base", num_labels=2)

model.to(globalconfig.device)
# model.to("cpu")

model.n_aug_targets = n_augs_dict






# # Run

from transformers import get_scheduler
from transformers import AdamW


# +
# define optimizer
### SGD as in paper
optimizer = torch.optim.SGD(model.parameters(), lr=globalconfig.lr, momentum=globalconfig.momentum, weight_decay=globalconfig.weight_decay)  # if not specified, the default lr is used

optimizer = AdamW(model.parameters(), lr=globalconfig.lr, weight_decay=globalconfig.weight_decay)




# create scheduler
num_training_steps = globalconfig.num_train_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_training_steps=num_training_steps,
)
# -



from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
globalconfig.grad_norm = 1.0

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f"../output/testMixup-{dataset}/TestTB-exp-n_{globalconfig.batch_size}-mixupAlpha_{globalconfig.alpha}-lr_{globalconfig.lr}-WeightDecay_{globalconfig.weight_decay}-Momentum_{globalconfig.momentum}-nEpochs_{globalconfig.num_train_epochs}")


# +
import logging

logger = logging.getLogger()

logging.basicConfig(
    filename="../log/mixupAug.txt",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)


# +
model.train()
for i_epoch in tqdm(range(globalconfig.num_train_epochs), file=open("../log/mixupTQDM.txt", 'w')):

    # loss_main_accum_avg = 0
    # loss_classification_accum = 0
    # loss_classification_accum_avg = 0
    # loss_mmd_accum = 0
    
    model.n_aug_tracker = {k:0 for k,v in n_augs_dict.items()}
    
    loss_accum_epoch = 0
    nSamples = 0
    
    loss_classification_accum_step = 0
    
    for i_step, batch in enumerate(train_loader):
        _GlobalStep = i_epoch * len(train_loader) + i_step

        reducedBatch = {k:v.to(globalconfig.device) for k,v in batch.items() if k not in ['Text']}

        ret = model(**reducedBatch, output_hidden_states=True)

        # target_domain = torch.LongTensor([z_category.index(x) for x in batch['domain_label']]).to(globalconfig.device)
        # target_primary = torch.LongTensor([y_Categories.index(x) for x in batch['primary_label']]).to(globalconfig.device)


        # loss_classification = criterion(ret["logits"], target_primary)
        
        
        nSamples += len(ret["logits"])
        
        writer.add_scalar("Loss_Main_step/train", ret["loss"], _GlobalStep)
        writer.add_scalar("LR", lr_scheduler.get_lr()[0], _GlobalStep)
        # update loss plot
        loss_accum_epoch += ret["loss"].item()
        

        # back propagate loss and clip gradients
        ret["loss"].backward()
        clip_grad_norm_(model.parameters(), globalconfig.grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        _success = _GlobalStep
        
    logger.info(model.n_aug_tracker)

    
    writer.add_scalar("Loss_Main_epoch/train", loss_accum_epoch/nSamples, i_epoch)
    writer.flush()


# -

reducedBatch

reducedBatch['domain_label'].shape

model.n_aug_targets

model.n_aug_tracker



model.save_pretrained(globalconfig.output_dir)

globalconfig.output_dir




