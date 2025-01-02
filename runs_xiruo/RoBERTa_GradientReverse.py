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
    default=3,
    help="Number of training epochs",
)
parser.add_argument("--batchSize", type=int, default=32, help="Batch size")
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="On which GPU to run",
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Directory to save outputs"
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
from transformers import AdamW
from transformers import get_scheduler


import itertools

from transformers import (
    AutoTokenizer,
)
from torch.autograd import Function
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


class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.model_name = "roberta-base"
globalconfig.max_seq_length = 512
globalconfig.device = args.device
globalconfig.batch_size = args.batchSize

globalconfig.lr = 1e-4
globalconfig.weight_decay = 1e-3
globalconfig.num_train_epochs = num_train_epochs

globalconfig.momentum = 0.9


# -


# +
globalconfig.output_dir = f"/bime-munin/xiruod/GradientReverse/{globalconfig.model_name}_{dataset}/n{n_test}/set-{pick_C}-epoch{globalconfig.num_train_epochs}"


# -


class TextDataset(Dataset):
    def __init__(self, df_in, txt_col, label_col, domain_col, tokenizer, max_length):
        self.primary_label = df_in[label_col]
        self.domain_label = df_in[domain_col].map({z_category[0]: 0, z_category[1]: 1})
        self.text = df_in[txt_col]
        self.tokenized = tokenizer(
            list(df_in[txt_col]),
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        self.tokenizer_keys = self.tokenized.keys()

    def __len__(self):
        return len(self.primary_label)

    def __getitem__(self, idx):
        primary_label = self.primary_label[idx]
        domain_label = self.domain_label[idx]
        text = self.text[idx]
        # input_ids = self.tokenized['input_ids'][idx]

        sample = {
            "Text": text,
            "primary_label": primary_label,
            "domain_label": domain_label,
        }
        for key in self.tokenizer_keys:
            sample[key] = self.tokenized[key][idx]

        return sample


tokenizer = AutoTokenizer.from_pretrained(globalconfig.model_name, use_fast=False)


# +
dataset_train = TextDataset(
    df_in=dfs["train"],
    txt_col=txt_col,
    label_col=df_split_label,
    domain_col=domain_col,
    tokenizer=tokenizer,
    max_length=globalconfig.max_seq_length,
)

train_loader = DataLoader(
    dataset_train, batch_size=globalconfig.batch_size, shuffle=True
)
# -


# # Model


from typing import List, Optional, Tuple, Union


from transformers import RobertaPreTrainedModel, RobertaModel

from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    SequenceClassifierOutput,
)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class RobertaForSequenceClassificationGradientReverse(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.classifierDomain = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

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
    ) -> Union[dict, Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
        logitsDomain = self.classifierDomain(grad_reverse(sequence_output))

        # loss = None
        # if labels is not None:
        #     # move labels to correct device to enable model parallelism
        #     labels = labels.to(logits.device)
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (
        #             labels.dtype == torch.long or labels.dtype == torch.int
        #         ):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = ()
        #         if self.num_labels == 1:
        #             loss = loss_fct(logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     elif self.config.problem_type == "multi_label_classification":
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(logits, labels)

        # lossDomain = loss_fct(logitsDomain, domain_label)

        return {
            # "lossMain": loss,
            # "lossDomain": lossDomain,
            "logitsMain": logits,
            "logitsDomain": logitsDomain,
        }

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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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


model = RobertaForSequenceClassificationGradientReverse.from_pretrained(
    "roberta-base", num_labels=2
)

model.to(globalconfig.device)
# model.to("cpu")


# # Run


# +

# # +
# # define optimizer
# ### SGD as in paper
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=globalconfig.lr,
    momentum=globalconfig.momentum,
    weight_decay=globalconfig.weight_decay,
)  # if not specified, the default lr is used

optimizer = AdamW(
    model.parameters(), lr=globalconfig.lr, weight_decay=globalconfig.weight_decay
)

# create scheduler
num_training_steps = globalconfig.num_train_epochs * len(train_loader)

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_training_steps=num_training_steps,
)
# -

criterion = CrossEntropyLoss(reduction="mean")  # , device=device))


from torch.nn.utils import clip_grad_norm_

globalconfig.grad_norm = 1.0


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(
    f"../output/testGradientReverse/{dataset}-set_{CombinationIdx}-TestTB-exp-n_{globalconfig.batch_size}-lr_{globalconfig.lr}-WeightDecay_{globalconfig.weight_decay}-Momentum_{globalconfig.momentum}-nEpochs_{globalconfig.num_train_epochs}"
)


# +
model.train()
for i_epoch in tqdm(
    range(globalconfig.num_train_epochs), file=open("../log/GradientReverse.txt", "w")
):

    loss_main_accum_avg = 0
    loss_main_accum = 0
    loss_main_accum_avg = 0
    loss_domain_accum = 0
    nSamples = 0

    loss_classification_accum_step = 0

    for i_step, batch in enumerate(train_loader):
        _GlobalStep = i_epoch * len(train_loader) + i_step

        reducedBatch = {
            k: v.to(globalconfig.device)
            for k, v in batch.items()
            if k not in ["Text", "primary_label", "domain_label"]
        }

        ret = model(**reducedBatch, output_hidden_states=True)
        target_domain = torch.LongTensor(
            batch["domain_label"]
        ).to(globalconfig.device)
        target_primary = torch.LongTensor(
            [y_Categories.index(x.item()) for x in batch["primary_label"]]
        ).to(globalconfig.device)

        loss_main = criterion(ret["logitsMain"], target_primary)
        loss_domain = criterion(ret["logitsDomain"], target_domain)

        nSamples += len(target_primary)

        writer.add_scalar("Loss_Main_step/train", loss_main, _GlobalStep)
        writer.add_scalar("Loss_Domain_step/train", loss_domain, _GlobalStep)

        writer.add_scalar("LR", lr_scheduler.get_lr()[0], _GlobalStep)

        ### accum by number of examples
        loss_main_accum += loss_main * len(target_primary)  # accum Main sum
        loss_domain_accum += loss_domain * len(target_primary)  # accum Domain norm

        # back propagate loss and clip gradients
        # self.loss_steps.append(loss.item())
        loss_main.backward(retain_graph=True)
        clip_grad_norm_(model.parameters(), globalconfig.grad_norm)

        loss_domain.backward()
        clip_grad_norm_(model.parameters(), globalconfig.grad_norm)

        # update loss plot
        # loss_epoch += loss.item()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        _success = _GlobalStep

    loss_main_accum_avg += loss_main_accum / nSamples
    loss_domain_accum_avg = loss_domain_accum / len(train_loader)

    # writer.add_scalar("Loss_Main_Accum/train", loss_main_accum, i_epoch)
    writer.add_scalar("Loss_Main_Accum_Avg/train", loss_main_accum_avg, i_epoch)

    writer.add_scalar("Loss_Domain_Accum_Avg/train", loss_domain_accum_avg, i_epoch)

    writer.flush()


# -


model.save_pretrained(globalconfig.output_dir)

globalconfig.output_dir
