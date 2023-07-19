import sys
import os
import random
import itertools
import pickle
import copy

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import number_split, create_mix
from src.backdoorBERT import backdoorAdjustBERTModel

# from src.data_process import load_wls_adress_AddDomain
from src.process_SHAC import load_process_SHAC


import warnings

warnings.simplefilter("ignore")
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
    v = 1
    n_valid_high = 10

    rand_seed = 34
    # rand_seed_np = 24
    # rand_seed_torch = 187

    num_labels = 2
    # zcol = None
    # zCats = None

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
    grad_reverse = False

    # assert zcol is not None
    # assert zCats is not None
    model_config = {}
    # model_config['model_type'] = model_type
    model_config["pretrained"] = pretrained
    # model_config["zcol"] = zcol
    # model_config['zCats'] = zCats
    model_config["max_length"] = max_length
    model_config["num_labels"] = num_labels
    model_config["hidden_dropout_prob"] = hidden_dropout_prob
    model_config["num_epochs"] = num_epochs
    model_config["num_warmup_steps"] = num_warmup_steps
    model_config["batch_size"] = batch_size
    model_config["lr"] = lr
    model_config["balance_weights"] = balance_weights
    model_config["grad_norm"] = grad_norm
    model_config["grad_reverse"] = grad_reverse

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
    alpha_test,
    train_test_ratio,
    n_test,
    n_valid_high,
    v,
    rand_seed,
    device,
):

    # ============= format
    p_mix_z1 = np.arange(p_mix_z1[0], p_mix_z1[1], p_mix_z1[2])
    alpha_test = np.arange(alpha_test[0], alpha_test[1], alpha_test[2])

    # ============= Load data
    df = load_process_SHAC(replaceNA="all")
    label = "Drug"
    txt_col = "text"
    z_Categories = ["uw", "mimic"]
    n_zCats = len(z_Categories)

    domain_col = "location"

    tmp = (
        pd.get_dummies(
            pd.Categorical(df[domain_col], categories=z_Categories), prefix="confounder"
        )
        * v
    )
    df = pd.concat([df, tmp], axis=1)

    df0 = df.query(f"location == '{z_Categories[0]}'").reset_index(drop=True)
    df1 = df.query(f"location == '{z_Categories[1]}'").reset_index(drop=True)

    confounder_cols = ["confounder_" + x for x in z_Categories]

    model_config = copy.deepcopy(model_config)
    model_config["zcol"] = domain_col
    model_config["n_zCats"] = n_zCats
    model_config["v"] = v

    # ============= calculate valid_high_combination and valid_full_settings
    theory_valid_full_settings = []
    for combination in itertools.product(
        p_pos_train_z0,  # [0.2],
        p_pos_train_z1,  # [0.4],
        p_mix_z1,  # np.arange(0.1, 0.999, 0.1),
        alpha_test,  # np.arange(0, 10, 0.1),
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

        if number_setting is not None:
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

    random.seed(rand_seed)
    rand_seed_np = random.randint(0, 9999)
    rand_seed_torch = random.randint(0, 9999)

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
            x_confounder_train = dfs["train"][confounder_cols]

            X_test = dfs["test"][txt_col]
            y_test = dfs["test"][[label]]
            x_confounder_test = dfs["test"][confounder_cols]

            p_z = []
            for _z in z_Categories:
                p_z.append(sum(dfs["train"][domain_col] == _z) / len(dfs["train"]))
            model_config["p_z"] = p_z

            # == Modeling Start
            model = backdoorAdjustBERTModel(**model_config)

            model.load_pretrained()

            # train & predict
            model.trainModel(X=X_train, y=y_train, z=x_confounder_train, device=device)

            y_main_prob = model.predict(X=X_test, z=x_confounder_test, device=device)

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
            y_test.to_csv(os.path.join(destination_runs, "y_test.csv"), index=False)

            y_main_prob.to_csv(
                os.path.join(destination_runs, "y_main_prob.csv"),
                index=False,
            )

            # save Epoch Loss Avg
            main_loss_epoch_avg = model.trainMainEpochLossAvg
            pd.DataFrame({"main_loss_epoch_avg": np.array(main_loss_epoch_avg)}).to_csv(
                os.path.join(destination_runs, "main_loss_epoch_avg.csv"), index=False
            )

            # TODO: move this out to make a standalone eval file
            # collect metrics: loss, auroc, auprc, f1
    #         loss = torch.nn.NLLLoss()

    #         _loss = loss(
    #             torch.log(torch.tensor(y_main_prob.values)),
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
    #                 y_score=y_main_prob,
    #                 average=average_curve,
    #             )
    #         )

    #         _auprc.append(
    #             metrics.average_precision_score(
    #                 y_true=multilabel_binarizer.fit_transform(y_test.values),
    #                 y_score=y_main_prob,
    #                 average=average_curve,
    #             )
    #         )
    #         _f1_at_05.append(
    #             metrics.f1_score(
    #                 y_true=y_test.values, y_pred=y_main_pred, average=average_f1
    #             )
    #         )

    #     losses_dict["losses"].append(losses)
    #     losses_dict["auroc"].append(_auroc)
    #     losses_dict["auprc"].append(_auprc)
    #     losses_dict["f1_at_05"].append(_f1_at_05)

    # # save metrics
    # with open(os.path.join(destination, "results.pkl"), "wb") as f:
    #     pickle.dump(obj=losses_dict, file=f)

    return 0
