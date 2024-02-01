########   NOTE: NOT FULL Settings!!!
# In this script, C_y is only limited to [0.2, 0.48, 0.72]


import sys
import os
import random
import itertools
import pickle
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import number_split, create_mix
from src.NeuralSingleLabelModel import NeuralSingleLabelModel

# from src.data_process import load_wls, load_adress
from src.process_SHAC import load_process_SHAC
import warnings
warnings.simplefilter('ignore')

                    
# Define experiment and load ingredients
ex = Experiment()


@ex.config
def cfg():

    proj_name = "Replace_with_a_NAME"

    p_pos_train_z0 = [0, 1, 0.1]
    p_pos_train_z1 = [0, 1, 0.2]
    p_mix_z1 = [0, 1, 0.05]  # will be changed into np.arange(0.1, 0.999, 0.1)
    
    numvals = 1023
    base = 1.1
    
    train_test_ratio = [4]
    n_test = [
        200
    ]  # the number of testing examples; set to None to disable (i.e., get as many examples as possible)

    n_valid_high = 10

    rand_seed_np = 24
    rand_seed_torch = 187

    num_labels = 2

    pretrained = "bert-base-uncased"
    device = "cuda:0"

    max_length = 512
    num_epochs = 3
    hidden_dropout_prob = 0.1
    num_warmup_steps = 0
    batch_size = 16

    lr = 1e-4
    grad_norm = 1.0
    balance_weights = False

    model_config = {}
    model_config["pretrained"] = pretrained
    model_config["max_length"] = max_length
    model_config["num_labels"] = num_labels
    model_config["hidden_dropout_prob"] = hidden_dropout_prob
    model_config["num_epochs"] = num_epochs
    model_config["num_warmup_steps"] = num_warmup_steps
    model_config["batch_size"] = batch_size
    model_config["lr"] = lr
    model_config["balance_weights"] = balance_weights
    model_config["grad_norm"] = grad_norm

    # Create observers
    destination = os.path.join("output", ex.path, proj_name)
    
    
    if not os.path.exists(destination):
        os.makedirs(destination)

    file_observ = FileStorageObserver.create(destination)
    ex.observers.append(file_observ)
    
    
    
@ex.automain
def main(
    model_config,
    destination,
    num_labels,
    p_pos_train_z0,
    p_pos_train_z1,
    p_mix_z1,
    numvals,
    base,
    train_test_ratio,
    n_test,
    n_valid_high,
    rand_seed_np,
    rand_seed_torch,
):

    # ============= format
   
    p_pos_train_z0 = np.arange(start=p_pos_train_z0[0], stop=p_pos_train_z0[1], step=p_pos_train_z0[2])
    p_pos_train_z1 = np.arange(start=p_pos_train_z1[0], stop=p_pos_train_z1[1], step=p_pos_train_z1[2])
    p_mix_z1 = np.arange(p_mix_z1[0], p_mix_z1[1], p_mix_z1[2])
    alpha_test_ls = np.power(base, np.arange(numvals))/np.power(base,numvals//2)
    
    # ============= Load data
    # df_wls_merge = load_wls()
    # df_adress = load_adress()
    df = load_process_SHAC(replaceNA='all')

    df0 = df.query("location == 'uw'").reset_index(drop=True)
    df1 = df.query("location == 'mimic'").reset_index(drop=True)
    label = "Drug"
    txt_col = "text"
    # ============= calculate valid_high_combination and valid_full_settings
    theory_valid_full_settings = []
    for combination in itertools.product(
        p_pos_train_z0,  # [0.2],
        p_pos_train_z1,  # [0.4],
        p_mix_z1,  # np.arange(0.1, 0.999, 0.1),
        alpha_test_ls,  # np.arange(0, 10, 0.1),
        train_test_ratio,  # [4],
        n_test,
    ):

        number_setting = number_split(
            p_pos_train_z0=combination[0],
            p_pos_train_z1=combination[1],
            p_mix_z1=combination[2],
            alpha_test=combination[3],
            train_test_ratio=combination[4],
            n_test=combination[5],
            verbose=False,
        )

        if (number_setting is not None) and (number_setting['mix_param_dict']['C_y'] in [0.2, 0.48, 0.72]):
            if np.all(
                [
                    number_setting[k] >= n_valid_high
                    for k in list(number_setting.keys())[:-1]
                ]
            ):
                theory_valid_full_settings.append(number_setting)

    # ============ Modeling
    losses_dict = {}

    losses_dict["combination"] = []
    losses_dict["full_setting"] = []
    losses_dict["losses"] = []
    # losses_dict["auroc"] = []
    # losses_dict["auprc"] = []
    # losses_dict["f1_at_05"] = []

    random.seed(rand_seed_np)
    np.random.seed(rand_seed_np)
    torch.manual_seed(rand_seed_torch)
    torch.cuda.manual_seed(rand_seed_torch)

    valid_n_full_settings = []

    for c in tqdm(theory_valid_full_settings):

        dfs = create_mix(
            df0=df0,
            df1=df1,
            target=label,
            setting=c,
            sample=False,
            seed=random.randint(0, 1000),
        )

        if dfs is None:
            continue
        valid_n_full_settings.append(c)

        destination_setting = os.path.join(
            destination,
            "setting_"
            + "_".join([f"{k}_{v:.4f}" for k, v in c["mix_param_dict"].items()]),
        )
        if not os.path.exists(destination_setting):
            os.makedirs(destination_setting)

        # save settings
        with open(os.path.join(destination_setting, "full_settings.pkl"), "wb") as f:
            pickle.dump(obj=c, file=f)

        losses_dict["combination"].append(c["mix_param_dict"])
        losses_dict["full_setting"].append(c)

        # losses = []
        # _auroc = []
        # _auprc = []
        # _f1_at_05 = []
        for i in range(5):

            dfs = create_mix(
                df0=df0,
                df1=df1,
                target=label,
                setting=c,
                sample=False,
                seed=random.randint(0, 1000),
            )

            assert dfs is not None
            
            X_train = dfs["train"][txt_col]
            y_train = dfs["train"][[label]]

            X_test = dfs["test"][txt_col]
            y_test = dfs["test"][[label]]

            model = NeuralSingleLabelModel(**model_config)

            model.load_pretrained()

            model.trainModel(X=X_train, y=y_train, device="cuda:0")

            y_pred, y_prob = model.predict(X=X_test, device="cuda:0")

            # save predictions
            destination_runs = os.path.join(
                destination_setting,
                f"RandomRun_{i}",
            )
            if not os.path.exists(destination_runs):
                os.makedirs(destination_runs)

            X_test.to_csv(
                os.path.join(destination_runs, "x_test.csv"),
                index=False,
            )
            y_test.to_csv(
                os.path.join(destination_runs, "y_test.csv"),
                index=False,
            )
            y_pred.to_csv(
                os.path.join(destination_runs, "y_pred.csv"),
                index=False,
            )
            y_prob.to_csv(
                os.path.join(destination_runs, "y_prob.csv"),
                index=False,
            )

            # save Epoch Loss Avg
            loss_epoch_avg = model.trainEpochLossAvg
            pd.DataFrame({"loss_epoch_avg": np.array(loss_epoch_avg)}).to_csv(
                os.path.join(destination_runs, "loss_epoch_avg.csv"), index=False
            )

    #         # NOTE: the following loss and auroc/auprc/f1 are only working for one column (hard-coded as "label") as y
    #         loss = torch.nn.NLLLoss()

    #         _loss = loss(
    #             torch.log(torch.tensor(y_prob.values)),
    #             torch.tensor(y_test.values).squeeze(1),
    #         )

    #         losses.append(_loss.item())

    #         # num_domain_label
    #         multilabel_binarizer = MultiLabelBinarizer(classes=range(num_labels))

    #         if num_labels == 2:
    #             average_curve = "macro"
    #             average_f1 = "binary"
    #         elif num_labels > 2:
    #             # ovr style AUROC/AUPRC, average="micro"
    #             average_curve = "micro"
    #             average_f1 = "micro"

    #         _auroc.append(
    #             metrics.roc_auc_score(
    #                 y_true=multilabel_binarizer.fit_transform(y_test.values),
    #                 y_score=y_prob,
    #                 average=average_curve,
    #             )
    #         )

    #         _auprc.append(
    #             metrics.average_precision_score(
    #                 y_true=multilabel_binarizer.fit_transform(y_test.values),
    #                 y_score=y_prob,
    #                 average=average_curve,
    #             )
    #         )
    #         _f1_at_05.append(
    #             metrics.f1_score(
    #                 y_true=y_test.values, y_pred=y_pred, average=average_f1
    #             )
    #         )

    #     losses_dict["losses"].append(losses)
    #     losses_dict["auroc"].append(_auroc)
    #     losses_dict["auprc"].append(_auprc)
    #     losses_dict["f1_at_05"].append(_f1_at_05)

    # with open(os.path.join(destination, "results.pkl"), "wb") as f:
    #     pickle.dump(obj=losses_dict, file=f)

    return 0