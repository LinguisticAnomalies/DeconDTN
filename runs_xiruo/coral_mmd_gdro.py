import os
import argparse


### Argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="SHAC",
    help="Dataset for the experiment",
)
parser.add_argument("-c", "--CombinationIdx", type=int, help="Set idx of c to use")
parser.add_argument("--nRuns", type=int, default=3, help="Number of experiments to run")
parser.add_argument(
    "--method", type=str, default="", choices=["GDRO", "MMD"], help="Directory to save outputs"
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Directory to save outputs"
)
parser.add_argument("--batchSize", type=int, default=32, help="Batch size")

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import sys


sys.path.append("../src")
sys.path.append("../config")

# +
from utils import number_split, create_mix, appendMetrics
import random
from copy import deepcopy
from tqdm import tqdm

# +
from transformers import (
    AutoTokenizer,
)

from torch.utils.data import Dataset, DataLoader

# -

from process_CD import load_cd
from process_HateSpeech import load_HateSpeech_dynGen, load_HateSpeech_wsf
from process_SHAC import load_process_SHAC
from sampling_numbers import HateSpeech_DICT, SHAC_DICT, CD_DICT


# +
# dataset = "HateSpeech"
dataset = args.dataset
CombinationIdx = args.CombinationIdx
# # CombinationIdx = 3636
# CombinationIdx = 6621


# dataset = "SHAC"
# CombinationIdx = 1152
# # CombinationIdx = 6114
# CombinationIdx = 11063
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

    # y_Categories = ["False", "True"]
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


dfs = create_mix(
    df0=df0,
    df1=df1,
    target=df_split_label,
    setting=c,
    sample=False,
    # seed=random.randint(0,1000),
    seed=222,
)

print("\nTraining...........\n")
class train_config:
    def __init__(self):
        self.quantization: bool = False


globalconfig = train_config()
globalconfig.model_name = "roberta-base"
globalconfig.max_seq_length = 512


globalconfig.lr = 1e-4
globalconfig.weight_decay = 1e-3
globalconfig.num_train_epochs = args.nRuns

globalconfig.momentum = 0.9
globalconfig.coral_lambda = -0.5

# globalconfig.device = "cpu"
# globalconfig.batch_size = 200 # for MMD


## for GDRO
globalconfig.device = args.device
globalconfig.batch_size = args.batchSize
globalconfig.num_train_epochs = args.nRuns

# -


class TextDataset(Dataset):
    def __init__(self, df_in, txt_col, label_col, domain_col, tokenizer, max_length):
        self.primary_label = df_in[label_col]
        self.domain_label = df_in[domain_col]
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

# +
# dataset_test = TextDataset(df_in=dfs['test'], txt_col=txt_col, label_col=label, domain_col=domain_col, tokenizer=tokenizer, max_length=globalconfig.max_seq_length)

# test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True)

# +
# for batch in train_loader:
#     break

# batch
# -


# # Auto Model

import torch

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    globalconfig.model_name, num_labels=len(y_Categories)
)

model.to(globalconfig.device)


# # # CORAL

# from transformers import get_scheduler
# from transformers import AdamW
# from torch.nn import CrossEntropyLoss


# # +
# # # define optimizer
# # ### AdamW

# # optimizer = AdamW(model.parameters(), lr=globalconfig.lr, weight_decay=globalconfig.weight_decay)

# # # create scheduler
# # num_training_steps = globalconfig.num_train_epochs * len(train_loader)

# # lr_scheduler = get_scheduler(
# #     "linear",
# #     optimizer=optimizer,
# #     num_warmup_steps=10,
# #     num_training_steps=num_training_steps,
# # )

# # +
# # define optimizer
# ### SGD as in paper
# optimizer = torch.optim.SGD(model.parameters(), lr=globalconfig.lr, momentum=globalconfig.momentum, weight_decay=globalconfig.weight_decay)  # if not specified, the default lr is used

# optimizer = AdamW(model.parameters(), lr=globalconfig.lr, weight_decay=globalconfig.weight_decay)


# # create scheduler
# num_training_steps = globalconfig.num_train_epochs * len(train_loader)

# lr_scheduler = get_scheduler(
#     "constant",
#     optimizer=optimizer,
#     num_training_steps=num_training_steps,
# )
# # -

# criterion = CrossEntropyLoss(reduction='mean') # , device=device))


# from tqdm import tqdm
# from torch.nn.utils import clip_grad_norm_
# globalconfig.grad_norm = 1.0

# # +
# # for i_step, batch in enumerate(tqdm(train_loader)):

# #     batch = {k: v.to(globalconfig.device) for k, v in batch.items()}

# #     ret = model(**batch)

# #     target_domain = torch.LongTensor([z_category.index(x) for x in batch['domain_label']])
# #     target_primary = torch.LongTensor([y_Categories.index(x) for x in batch['primary_label']])


# #     loss_primary = criterion(ret["outputs_main_classifier"], target_primary)

# #     covariance_matrices = []
# #     for _z in range(len(z_category)):
# #         _sourcePooler = ret['pooler'][(target_domain == _z).type(torch.bool)]
# #         _cov = torch.cov(_sourcePooler.T)

# #         covariance_matrices.append(_cov)

# #     if len(covariance_matrices) == 1:
# #         continue
# #     else:
# #         ### Assume there are only 2 categories for now!
# #         assert len(covariance_matrices) == 2
# #         loss_frob = torch.square(torch.norm(covariance_matrices[0]-covariance_matrices[1], p='fro'))/4/(base_config.hidden_size**2)

# #     loss_main = loss_primary + loss_frob

# #     # back propagate loss and clip gradients
# #     # self.loss_steps.append(loss.item())
# #     loss_main.backward()
# #     clip_grad_norm_(model.parameters(), globalconfig.grad_norm)

# #     # update loss plot
# #     # loss_epoch += loss.item()

# #     optimizer.step()
# #     lr_scheduler.step()
# #     optimizer.zero_grad()
# #     break
# # -

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(f"../output/test/{dataset}-set_{CombinationIdx}-TestTB-exp-n_{globalconfig.batch_size}-coralLambda_{globalconfig.coral_lambda}-lr_{globalconfig.lr}-WeightDecay_{globalconfig.weight_decay}-Momentum_{globalconfig.momentum}-nEpochs_{globalconfig.num_train_epochs}")


# f"../output/test/TestTB-exp-n_{globalconfig.batch_size}-coralLambda_{globalconfig.coral_lambda}-lr_{globalconfig.lr}-WeightDecay_{globalconfig.weight_decay}-Momentum_{globalconfig.momentum}-nEpochs_{globalconfig.num_train_epochs}"


# # +
# model.train()
# for i_epoch in range(globalconfig.num_train_epochs):

#     loss_main_accum_avg = 0
#     loss_classification_accum = 0
#     loss_classification_accum_avg = 0
#     loss_frob_accum = 0
#     nSamples = 0

#     loss_classification_accum_step = 0

#     for i_step, batch in enumerate(tqdm(train_loader)):
#         _GlobalStep = i_epoch * len(train_loader) + i_step

#         reducedBatch = {k:v.to(globalconfig.device) for k,v in batch.items() if k not in ['Text', 'primary_label', 'domain_label']}

#         # batch = {k: v.to(globalconfig.device) for k, v in batch.items()}

#         ret = model(**reducedBatch, output_hidden_states=True)

#         target_domain = torch.LongTensor([z_category.index(x) for x in batch['domain_label']]).to(globalconfig.device)
#         target_primary = torch.LongTensor([y_Categories.index(x) for x in batch['primary_label']]).to(globalconfig.device)


#         loss_classification = criterion(ret["logits"], target_primary)
#         covariance_matrices = []
#         for _z in range(len(z_category)):
#             # get pooler: last layer of hidden states (-1), of the CLS token ([:,0,:])
#             _sourcePooler = ret['hidden_states'][-1][:, 0, :][(target_domain == _z).type(torch.bool)]
#             _cov = torch.cov(_sourcePooler.T)

#             covariance_matrices.append(_cov)

#         if len(covariance_matrices) == 1:  # in case there is only one domain category in the batch

#             loss_main = loss_classification
#             writer.add_scalar("Loss_Frob/train", np.NaN, _GlobalStep)
#             loss_frob = 0

#         else:
#             ### Assume there are only 2 categories for now!
#             assert len(covariance_matrices) == 2
#             loss_frob = torch.square(torch.norm(covariance_matrices[0]-covariance_matrices[1], p='fro'))/4/(model.config.hidden_size**2)
#             writer.add_scalar("Loss_Frob_step/train", loss_frob, _GlobalStep)

#             loss_main = loss_classification + globalconfig.coral_lambda * loss_frob


#         nSamples += len(target_primary)

#         writer.add_scalar("Loss_Main_step/train", loss_main, _GlobalStep)
#         writer.add_scalar("Loss_Classification_step/train", loss_classification, _GlobalStep)
#         writer.add_scalar("LR", lr_scheduler.get_lr()[0], _GlobalStep)


#         ### accum by number of examples
#         loss_classification_accum += loss_classification * len(target_primary)  # accum sum
#         loss_frob_accum += loss_frob  # accum Frob norm

#         ### accum by step
#         loss_classification_accum_step += loss_classification


#         # back propagate loss and clip gradients
#         # self.loss_steps.append(loss.item())
#         loss_main.backward()
#         clip_grad_norm_(model.parameters(), globalconfig.grad_norm)

#         # update loss plot
#         # loss_epoch += loss.item()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

#         _success = _GlobalStep

#     loss_classification_accum_avg += loss_classification_accum / nSamples
#     loss_frob_accum_avg = loss_frob_accum/len(train_loader)
#     loss_main_accum_avg += loss_classification_accum_avg + loss_frob_accum_avg


#     # writer.add_scalar("Loss_Main_Accum/train", loss_main_accum, i_epoch)
#     writer.add_scalar("Loss_Main_Accum_Avg/train", loss_main_accum_avg, i_epoch)
#     writer.add_scalar("Loss_Classification_Accum/train", loss_classification_accum, i_epoch)
#     writer.add_scalar("Loss_Classification_Accum_Avg/train", loss_classification_accum_avg, i_epoch)
#     writer.add_scalar("Loss_Frob_Accum/train", loss_frob_accum, i_epoch)
#     writer.add_scalar("Loss_Frob_Accum_Avg/train", loss_frob_accum_avg, i_epoch)

#     writer.add_scalar("Loss_Classification_Accum_Avg_ByStep/train", loss_classification_accum_step/len(train_loader), i_epoch)


# # -


# len(train_loader)

# num_training_steps

# nSamples


# globalconfig.output_dir = "../output/test/roberta/"

# # +

# output_dir = globalconfig.output_dir
# os.makedirs(output_dir, exist_ok=True)

# # outfile = f"{output_dir}/set-{args.CombinationIdx}-epoch{globalconfig.num_train_epochs}.pth"
# outfile = f"{output_dir}/ttt.pth"

# torch.save(model, outfile)
# # -


# loss_frob

if args.method == "MMD":
    # # MMD Loss

    # +
    import torch
    from torch import nn


    class RBF(nn.Module):

        def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
            super().__init__()
            self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
            self.bandwidth = bandwidth

        def get_bandwidth(self, L2_distances):
            if self.bandwidth is None:
                n_samples = L2_distances.shape[0]
                return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

            return self.bandwidth

        def forward(self, X):
            L2_distances = torch.cdist(X, X) ** 2
            return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


    class MMDLoss(nn.Module):

        def __init__(self, kernel=RBF()):
            super().__init__()
            self.kernel = kernel

        def forward(self, X, Y):
            K = self.kernel(torch.vstack([X, Y]))

            X_size = X.shape[0]
            XX = K[:X_size, :X_size].mean()
            XY = K[:X_size, X_size:].mean()
            YY = K[X_size:, X_size:].mean()
            return XX - 2 * XY + YY


    # -

    from transformers import get_scheduler
    from transformers import AdamW
    from torch.nn import CrossEntropyLoss


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

    criterion = CrossEntropyLoss(reduction='mean') # , device=device))


    from tqdm import tqdm
    from torch.nn.utils import clip_grad_norm_
    globalconfig.grad_norm = 1.0

    globalconfig.mmd_lambda = 0.25


    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"../output/testMMD/{dataset}-set_{CombinationIdx}-TestTB-exp-n_{globalconfig.batch_size}-MMDLambda_{globalconfig.mmd_lambda}-lr_{globalconfig.lr}-WeightDecay_{globalconfig.weight_decay}-Momentum_{globalconfig.momentum}-nEpochs_{globalconfig.num_train_epochs}")


    # +
    model.train()
    for i_epoch in tqdm(range(globalconfig.num_train_epochs), file=open("../log/MMD.txt", "w")):

        loss_main_accum_avg = 0
        loss_classification_accum = 0
        loss_classification_accum_avg = 0
        loss_mmd_accum = 0
        nSamples = 0

        loss_classification_accum_step = 0

        for i_step, batch in enumerate(train_loader):
            _GlobalStep = i_epoch * len(train_loader) + i_step

            reducedBatch = {k:v.to(globalconfig.device) for k,v in batch.items() if k not in ['Text', 'primary_label', 'domain_label']}

            # batch = {k: v.to(globalconfig.device) for k, v in batch.items()}

            ret = model(**reducedBatch, output_hidden_states=True)

            target_domain = torch.LongTensor([z_category.index(x) for x in batch['domain_label']]).to(globalconfig.device)
            target_primary = torch.LongTensor([y_Categories.index(x) for x in batch['primary_label']]).to(globalconfig.device)


            loss_classification = criterion(ret["logits"], target_primary)

            criterionMMD = MMDLoss()

            lastHidden_ls = []
            for _z in range(len(z_category)):
                # get pooler: last layer of hidden states (-1), of the CLS token ([:,0,:])
                _sourcePooler = ret['hidden_states'][-1][:, 0, :][(target_domain == _z).type(torch.bool)]

                lastHidden_ls.append(_sourcePooler)

            if len(lastHidden_ls) == 1:
                loss_main = loss_classification
                writer.add_scalar("Loss_MMD_step/train", np.NaN, _GlobalStep)
            else:
                loss_MMD = criterionMMD(X=lastHidden_ls[0], Y=lastHidden_ls[1])
                writer.add_scalar("Loss_MMD_step/train", loss_MMD, _GlobalStep)

                loss_main = loss_classification + globalconfig.mmd_lambda * loss_MMD


            nSamples += len(target_primary)

            writer.add_scalar("Loss_Main_step/train", loss_main, _GlobalStep)
            writer.add_scalar("Loss_Classification_step/train", loss_classification, _GlobalStep)
            writer.add_scalar("LR", lr_scheduler.get_lr()[0], _GlobalStep)


            ### accum by number of examples
            loss_classification_accum += loss_classification * len(target_primary)  # accum sum
            loss_mmd_accum += loss_MMD  # accum MMD norm

            ### accum by step
            loss_classification_accum_step += loss_classification


            # back propagate loss and clip gradients
            # self.loss_steps.append(loss.item())
            loss_main.backward()
            clip_grad_norm_(model.parameters(), globalconfig.grad_norm)

            # update loss plot
            # loss_epoch += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            _success = _GlobalStep

        loss_classification_accum_avg += loss_classification_accum / nSamples
        loss_mmd_accum_avg = loss_mmd_accum/len(train_loader)
        loss_main_accum_avg += loss_classification_accum_avg + loss_mmd_accum_avg


        # writer.add_scalar("Loss_Main_Accum/train", loss_main_accum, i_epoch)
        writer.add_scalar("Loss_Main_Accum_Avg/train", loss_main_accum_avg, i_epoch)
        writer.add_scalar("Loss_Classification_Accum/train", loss_classification_accum, i_epoch)
        writer.add_scalar("Loss_Classification_Accum_Avg/train", loss_classification_accum_avg, i_epoch)
        writer.add_scalar("Loss_MMD_Accum/train", loss_mmd_accum, i_epoch)
        writer.add_scalar("Loss_MMD_Accum_Avg/train", loss_mmd_accum_avg, i_epoch)

        writer.add_scalar("Loss_Classification_Accum_Avg_ByStep/train", loss_classification_accum_step/len(train_loader), i_epoch)

        writer.flush()


    # -

    y_Categories

    _success

    # +
    globalconfig.output_dir = f"/bime-munin/xiruod/MMD/{globalconfig.model_name}_{dataset}/n{n_test}/set-{pick_C}-epoch{globalconfig.num_train_epochs}"


    # +

    output_dir = globalconfig.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # outfile = f"{output_dir}/set-{args.CombinationIdx}-epoch{globalconfig.num_train_epochs}.pth"
    # outfile = f"{output_dir}/ttt.pth"

    model.save_pretrained(globalconfig.output_dir)

    # torch.save(model, outfile)
    # -

elif args.method == "GDRO":




    # # GDRO Loss

    # +
    group_counts = []

    for _ in z_category:
        group_counts.append(sum(dfs["train"][domain_col] == _))
    # -

    torch.tensor(group_counts)

    # +
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np


    class LossComputer:
        def __init__(
            self,
            criterion,
            is_robust,
            group_counts,
            alpha=None,
            gamma=0.1,
            adj=None,
            min_var_weight=0,
            step_size=0.01,
            normalize_loss=False,
            btl=False,
        ):
            self.criterion = criterion
            self.is_robust = is_robust
            self.gamma = gamma
            self.alpha = alpha
            self.min_var_weight = min_var_weight
            self.step_size = step_size
            self.normalize_loss = normalize_loss
            self.btl = btl

            # self.n_groups = dataset.n_groups
            # self.group_counts = dataset.group_counts().to(globalconfig.device)
            self.n_groups = len(group_counts)
            self.group_counts = torch.tensor(group_counts).to(globalconfig.device)
            self.group_frac = self.group_counts / self.group_counts.sum()

            if adj is not None:
                self.adj = torch.from_numpy(adj).float().to(globalconfig.device)
            else:
                self.adj = torch.zeros(self.n_groups).float().to(globalconfig.device)

            if is_robust:
                assert alpha, "alpha must be specified"

            # quantities maintained throughout training
            self.adv_probs = (
                torch.ones(self.n_groups).to(globalconfig.device) / self.n_groups
            )
            self.exp_avg_loss = torch.zeros(self.n_groups).to(globalconfig.device)
            self.exp_avg_initialized = (
                torch.zeros(self.n_groups).byte().to(globalconfig.device)
            )

            self.reset_stats()

        def loss(self, yhat, y, group_idx=None, is_training=False):
            # compute per-sample and per-group losses
            per_sample_losses = self.criterion(yhat, y)
            group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
            group_acc, group_count = self.compute_group_avg(
                (torch.argmax(yhat, 1) == y).float(), group_idx
            )

            # update historical losses
            self.update_exp_avg_loss(group_loss, group_count)

            # compute overall loss
            if self.is_robust and not self.btl:
                actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
            elif self.is_robust and self.btl:
                actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
            else:
                actual_loss = per_sample_losses.mean()
                weights = None

            # update stats
            self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

            return actual_loss

        def compute_robust_loss(self, group_loss, group_count):
            adjusted_loss = group_loss
            if torch.all(self.adj > 0):
                adjusted_loss += self.adj / torch.sqrt(self.group_counts)
            if self.normalize_loss:
                adjusted_loss = adjusted_loss / (adjusted_loss.sum())
            self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
            self.adv_probs = self.adv_probs / (self.adv_probs.sum())

            robust_loss = group_loss @ self.adv_probs
            return robust_loss, self.adv_probs

        def compute_robust_loss_btl(self, group_loss, group_count):
            adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
            return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

        def compute_robust_loss_greedy(self, group_loss, ref_loss):
            sorted_idx = ref_loss.sort(descending=True)[1]
            sorted_loss = group_loss[sorted_idx]
            sorted_frac = self.group_frac[sorted_idx]

            mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
            weights = mask.float() * sorted_frac / self.alpha
            last_idx = mask.sum()
            weights[last_idx] = 1 - weights.sum()
            weights = sorted_frac * self.min_var_weight + weights * (
                1 - self.min_var_weight
            )

            robust_loss = sorted_loss @ weights

            # sort the weights back
            _, unsort_idx = sorted_idx.sort()
            unsorted_weights = weights[unsort_idx]
            return robust_loss, unsorted_weights

        def compute_group_avg(self, losses, group_idx):
            # compute observed counts and mean loss for each group
            group_map = (
                group_idx
                == torch.arange(self.n_groups).unsqueeze(1).long().to(globalconfig.device)
            ).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count == 0).float()  # avoid nans
            group_loss = (group_map @ losses.view(-1)) / group_denom
            return group_loss, group_count

        def update_exp_avg_loss(self, group_loss, group_count):
            prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
                self.exp_avg_initialized > 0
            ).float()
            curr_weights = 1 - prev_weights
            self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
            self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

        def reset_stats(self):
            self.processed_data_counts = torch.zeros(self.n_groups).to(globalconfig.device)
            self.update_data_counts = torch.zeros(self.n_groups).to(globalconfig.device)
            self.update_batch_counts = torch.zeros(self.n_groups).to(globalconfig.device)
            self.avg_group_loss = torch.zeros(self.n_groups).to(globalconfig.device)
            self.avg_group_acc = torch.zeros(self.n_groups).to(globalconfig.device)
            self.avg_per_sample_loss = 0.0
            self.avg_actual_loss = 0.0
            self.avg_acc = 0.0
            self.batch_count = 0.0

        def update_stats(
            self, actual_loss, group_loss, group_acc, group_count, weights=None
        ):
            # avg group loss
            denom = self.processed_data_counts + group_count
            denom += (denom == 0).float()
            prev_weight = self.processed_data_counts / denom
            curr_weight = group_count / denom
            self.avg_group_loss = (
                prev_weight * self.avg_group_loss + curr_weight * group_loss
            )

            # avg group acc
            self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

            # batch-wise average actual loss
            denom = self.batch_count + 1
            self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (
                1 / denom
            ) * actual_loss

            # counts
            self.processed_data_counts += group_count
            if self.is_robust:
                self.update_data_counts += group_count * ((weights > 0).float())
                self.update_batch_counts += ((group_count * weights) > 0).float()
            else:
                self.update_data_counts += group_count
                self.update_batch_counts += (group_count > 0).float()
            self.batch_count += 1

            # avg per-sample quantities
            group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
            self.avg_per_sample_loss = group_frac @ self.avg_group_loss
            self.avg_acc = group_frac @ self.avg_group_acc

        def get_model_stats(self, model, args, stats_dict):
            model_norm_sq = 0.0
            for param in model.parameters():
                model_norm_sq += torch.norm(param) ** 2
            stats_dict["model_norm_sq"] = model_norm_sq.item()
            stats_dict["reg_loss"] = args.weight_decay / 2 * model_norm_sq.item()
            return stats_dict

        def get_stats(self, model=None, args=None):
            stats_dict = {}
            for idx in range(self.n_groups):
                stats_dict[f"avg_loss_group:{idx}"] = self.avg_group_loss[idx].item()
                stats_dict[f"exp_avg_loss_group:{idx}"] = self.exp_avg_loss[idx].item()
                stats_dict[f"avg_acc_group:{idx}"] = self.avg_group_acc[idx].item()
                stats_dict[f"processed_data_count_group:{idx}"] = (
                    self.processed_data_counts[idx].item()
                )
                stats_dict[f"update_data_count_group:{idx}"] = self.update_data_counts[
                    idx
                ].item()
                stats_dict[f"update_batch_count_group:{idx}"] = self.update_batch_counts[
                    idx
                ].item()

            stats_dict["avg_actual_loss"] = self.avg_actual_loss.item()
            stats_dict["avg_per_sample_loss"] = self.avg_per_sample_loss.item()
            stats_dict["avg_acc"] = self.avg_acc.item()

            # Model stats
            if model is not None:
                assert args is not None
                stats_dict = self.get_model_stats(model, args, stats_dict)

            return stats_dict


    # -

    from transformers import get_scheduler
    from transformers import AdamW
    from torch.nn import CrossEntropyLoss

    # +
    # define optimizer
    ### SGD as in paper
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

    criterionFull = CrossEntropyLoss(reduction="none")  # , device=device))


    train_loss_computer = LossComputer(
        criterionFull,
        is_robust=True,
        group_counts=group_counts,
        alpha=0.2,
        gamma=0.1,
        adj=np.array([0]),
        step_size=0.01,
        normalize_loss=False,
        btl=False,
        min_var_weight=0.1,
    )

    from torch.nn.utils import clip_grad_norm_

    globalconfig.grad_norm = 1.0

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(
        f"../output/testGDRO/{dataset}-set_{CombinationIdx}-TestTB-exp-n_{globalconfig.batch_size}-lr_{globalconfig.lr}-WeightDecay_{globalconfig.weight_decay}-Momentum_{globalconfig.momentum}-nEpochs_{globalconfig.num_train_epochs}"
    )

    # +
    model.train()
    for i_epoch in tqdm(
        range(globalconfig.num_train_epochs), file=open("../log/GDRO.txt", "w")
    ):

        loss_classification_accum = 0
        loss_classification_accum_avg = 0
        nSamples = 0

        loss_classification_accum_step = 0

        for i_step, batch in enumerate(train_loader):
            _GlobalStep = i_epoch * len(train_loader) + i_step

            reducedBatch = {
                k: v.to(globalconfig.device)
                for k, v in batch.items()
                if k not in ["Text", "primary_label", "domain_label"]
            }

            # batch = {k: v.to(globalconfig.device) for k, v in batch.items()}

            ret = model(**reducedBatch, output_hidden_states=True)

            target_primary = torch.LongTensor(
                [y_Categories.index(x) for x in batch["primary_label"]]
            ).to(globalconfig.device)
            target_domain = torch.LongTensor(
                [z_category.index(x) for x in batch["domain_label"]]
            ).to(globalconfig.device)

            #         idx_c0 = torch.tensor(np.array(batch['domain_label']) == z_category[0], dtype=torch.bool)
            #         idx_c1 = torch.tensor(np.array(batch['domain_label']) == z_category[1], dtype=torch.bool)

            #         loss_cat0 = criterion(ret['logits'][idx_c0], target_primary[idx_c0])
            #         loss_cat1 = criterion(ret['logits'][idx_c1], target_primary[idx_c1])

            #         loss_classification = torch.maximum(loss_cat0, loss_cat1)
            loss_classification = train_loss_computer.loss(
                ret["logits"], target_primary, target_domain, True
            )

            nSamples += len(target_primary)

            writer.add_scalar(
                "Loss_Classification_step/train", loss_classification, _GlobalStep
            )
            writer.add_scalar("LR", lr_scheduler.get_lr()[0], _GlobalStep)

            ### accum by number of examples
            loss_classification_accum += loss_classification * len(
                target_primary
            )  # accum sum

            ### accum by step
            loss_classification_accum_step += loss_classification

            # back propagate loss and clip gradients
            # self.loss_steps.append(loss.item())
            loss_classification.backward()
            clip_grad_norm_(model.parameters(), globalconfig.grad_norm)

            # update loss plot
            # loss_epoch += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            _success = _GlobalStep

        loss_classification_accum_avg += loss_classification_accum / nSamples

        writer.add_scalar(
            "Loss_Classification_Accum/train", loss_classification_accum, i_epoch
        )
        writer.add_scalar(
            "Loss_Classification_Accum_Avg/train", loss_classification_accum_avg, i_epoch
        )

        writer.add_scalar(
            "Loss_Classification_Accum_Avg_ByStep/train",
            loss_classification_accum_step / len(train_loader),
            i_epoch,
        )

        writer.flush()


    # +
    globalconfig.output_dir = f"/bime-munin/xiruod/GDRO/{globalconfig.model_name}_{dataset}/n{n_test}/set-{pick_C}-epoch{globalconfig.num_train_epochs}"


    # +

    output_dir = globalconfig.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # outfile = f"{output_dir}/set-{args.CombinationIdx}-epoch{globalconfig.num_train_epochs}.pth"
    # outfile = f"{output_dir}/ttt.pth"

    model.save_pretrained(globalconfig.output_dir)

    # torch.save(model, outfile)
    # -

    globalconfig.output_dir

    
    
    
    
print("\nFinished Training...........\n")
