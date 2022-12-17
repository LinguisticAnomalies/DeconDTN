import sys
import os
import random
import itertools
import pickle
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import confoundSplitNumbers, confoundSplitDF
from src.AdversarialModel import GradientReverseModel
from src.data_process import load_wls_adress_AddDomain


# Define experiment and load ingredients
ex = Experiment()


@ex.config
def cfg():

    proj_name = "exp_02_04"

    p_pos_train_z0 = [0.2]
    p_pos_train_z1 = [0.4]
    p_mix_z1 = [0.1, 0.999, 0.1]  # will be changed into np.arange(0.1, 0.999, 0.1)
    alpha_test = [0, 10, 0.1]  # np.arange(0, 10, 0.1)
    train_test_ratio = [4]
    n_test = [
        150
    ]  # the number of testing examples; set to None to disable (i.e., get as many examples as possible)
    n_test_error = [0]

    n_valid_high = 10

    rand_seed_np = 24
    rand_seed_torch = 187

    num_labels = 2
    num_domain_labels = 2

    pretrained = "bert-base-uncased"
    device = "cuda:0"

    max_length = 120
    num_epochs = 6
    hidden_dropout_prob = 0.1
    num_warmup_steps = 0
    batch_size = 50

    lr = 1e-5
    grad_norm = 1.0
    balance_weights = False

    model_config = {}
    # model_config['model_type'] = model_type
    model_config["pretrained"] = pretrained
    model_config["max_length"] = max_length
    model_config["num_labels"] = num_labels
    model_config["num_domain_labels"] = num_domain_labels
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
    num_domain_labels,
    p_pos_train_z0,
    p_pos_train_z1,
    p_mix_z1,
    alpha_test,
    train_test_ratio,
    n_test,
    n_test_error,
    n_valid_high,
    rand_seed_np,
    rand_seed_torch,
):

    # ============= format
    p_mix_z1 = np.arange(p_mix_z1[0], p_mix_z1[1], p_mix_z1[2])
    alpha_test = np.arange(alpha_test[0], alpha_test[1], alpha_test[2])

    # ============= Load data
    df_wls_merge = load_wls_adress_AddDomain(dt="wls")
    df_adress = load_wls_adress_AddDomain(dt="adress")

    # ============= calculate valid_high_combination and valid_full_settings
    valid_high_combinations = []
    valid_full_settings = []

    for combination in itertools.product(
        p_pos_train_z0,  # [0.2],
        p_pos_train_z1,  # [0.4],
        p_mix_z1,  # np.arange(0.1, 0.999, 0.1),
        alpha_test,  # np.arange(0, 10, 0.1),
        train_test_ratio,  # [4],
        n_test,
        n_test_error,
    ):
        ret = confoundSplitNumbers(
            df0=df_wls_merge,
            df1=df_adress,
            df0_label="label",
            df1_label="label",
            p_pos_train_z0=combination[0],
            p_pos_train_z1=combination[1],
            p_mix_z1=combination[2],
            alpha_test=combination[3],
            train_test_ratio=combination[4],
            n_test=combination[5],
            n_test_error=combination[6],
        )

        if (ret is not None) and (
            ret["n_df0_train_pos"] >= n_valid_high
        ):  # valie high combos
            valid_high_combinations.append(combination)
            valid_full_settings.append(ret)

    # ============ Modeling
    losses_dict = {}

    losses_dict["combination"] = []
    losses_dict["full_setting"] = []
    losses_dict["losses"] = []
    losses_dict["auroc"] = []
    losses_dict["auprc"] = []
    losses_dict["f1_at_05"] = []

    random.seed(rand_seed_np)
    np.random.seed(rand_seed_np)
    torch.manual_seed(rand_seed_torch)
    torch.cuda.manual_seed(rand_seed_torch)

    for c, setting in tqdm(
        zip(valid_high_combinations, valid_full_settings),
        total=len(valid_high_combinations),
    ):
        losses_dict["combination"].append(c)
        losses_dict["full_setting"].append(setting)

        losses = []
        _auroc = []
        _auprc = []
        _f1_at_05 = []

        for i in range(5):
            _rand = random.randint(0, 1000)

            # Given combination of parameters, find dataframe split
            combination = c
            # combination = (0.201, 0.6, 0.3, 1.0, 4)

            ret = confoundSplitDF(
                df0=df_wls_merge,
                df1=df_adress,
                df0_label="label",
                df1_label="label",
                p_pos_train_z0=combination[0],
                p_pos_train_z1=combination[1],
                p_mix_z1=combination[2],
                alpha_test=combination[3],
                train_test_ratio=combination[4],
                random_state=_rand,
                n_test=combination[5],
                n_test_error=combination[6],
            )

            # split dataframe
            df_train = pd.concat(
                [
                    ret["sample_df0_train"][["text", "label", "domain_index"]],
                    ret["sample_df1_train"][["text", "label", "domain_index"]],
                ],
                ignore_index=True,
            )

            df_test = pd.concat(
                [
                    ret["sample_df0_test"][["text", "label", "domain_index"]],
                    ret["sample_df1_test"][["text", "label", "domain_index"]],
                ],
                ignore_index=True,
            )

            X_train = df_train["text"]
            y_train = df_train[["label"]]
            y_domain_train = df_train[["domain_index"]]

            X_test = df_test["text"]
            y_test = df_test[["label"]]
            y_domain_test = df_test[["domain_index"]]

            # == Modeling Start
            model = GradientReverseModel(**model_config)

            model.load_pretrained()

            # train & predict
            model.trainModel(
                X=X_train, y=y_train, y_domain=y_domain_train, device="cuda:0"
            )

            y_main_pred, y_main_prob, y_domain_pred, y_domain_prob = model.predict(
                X=X_test, device="cuda:0"
            )

            # save predictions
            destination_runs = os.path.join(
                destination, "_".join([str(x) for x in c]), f"RandomRun_{i}"
            )
            if not os.path.exists(destination_runs):
                os.makedirs(destination_runs)

            X_test.to_csv(
                os.path.join(destination_runs, "x_test.csv"),
                index=False,
            )
            y_main_pred.to_csv(
                os.path.join(destination_runs, "y_main_pred.csv"),
                index=False,
            )
            y_main_prob.to_csv(
                os.path.join(destination_runs, "y_main_prob.csv"),
                index=False,
            )
            y_domain_pred.to_csv(
                os.path.join(destination_runs, "y_domain_pred.csv"),
                index=False,
            )
            y_domain_prob.to_csv(
                os.path.join(destination_runs, "y_domain_prob.csv"),
                index=False,
            )

            # collect metrics: loss, auroc, auprc, f1
            loss = torch.nn.NLLLoss()

            _loss = loss(
                torch.log(torch.tensor(y_main_prob.values)),
                torch.tensor(y_test.values).squeeze(1),
            )

            losses.append(_loss.item())

            # num_domain_label
            multilabel_binarizer = MultiLabelBinarizer(classes=range(num_labels))

            if num_labels == 2:
                average_curve = "macro"
                average_f1 = "binary"
            elif num_labels > 2:
                # ovr style AUROC/AUPRC, average="micro"
                average_curve = "micro"
                average_f1 = "micro"

            _auroc.append(
                metrics.roc_auc_score(
                    y_true=multilabel_binarizer.fit_transform(y_test.values),
                    y_score=y_main_prob,
                    average=average_curve,
                )
            )

            _auprc.append(
                metrics.average_precision_score(
                    y_true=multilabel_binarizer.fit_transform(y_test.values),
                    y_score=y_main_prob,
                    average=average_curve,
                )
            )
            _f1_at_05.append(
                metrics.f1_score(
                    y_true=y_test.values, y_pred=y_main_pred, average=average_f1
                )
            )

        losses_dict["losses"].append(losses)
        losses_dict["auroc"].append(_auroc)
        losses_dict["auprc"].append(_auprc)
        losses_dict["f1_at_05"].append(_f1_at_05)

    # save metrics
    with open(os.path.join(destination, "results.pkl"), "wb") as f:
        pickle.dump(obj=losses_dict, file=f)

    return 0
