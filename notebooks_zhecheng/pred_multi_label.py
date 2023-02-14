#!/usr/bin/env python3

import sys
import os
import random

import itertools
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import torch




sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.MultiLabel import MultiLabel

# define ingredient and experiments



    
def main(
    rand_seed_np = 2022,
    rand_seed_torch = 2022):
    destination = '/home/sheng136/workspace/deDTN/code/DeconDTN/runs_zhecheng/output'
    
    num_labels = 2
    
    pretrained = "bert-base-uncased"
    device = "cuda:0"

    max_length = 120
    num_epochs = 6
    #problem_type = "multi_classification"
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
    model_config["hidden_dropout_prob"] = hidden_dropout_prob
    model_config["num_epochs"] = num_epochs
    model_config["num_warmup_steps"] = num_warmup_steps
    model_config["batch_size"] = batch_size
    model_config["lr"] = lr
    model_config["balance_weights"] = balance_weights
    model_config["grad_norm"] = grad_norm
    
    
    
    # load data
    df = pd.read_csv('/home/sheng136/workspace/deDTN/code/DeconDTN/notebooks_zhecheng/processed_db.csv')
    
    df = df.dropna(subset = ['gender','text'], how = 'any')
    
    # label encoder
    le = LabelEncoder()
    df['source'] = le.fit_transform(df['source'])

    #df['gender'] = df['gender'].map({2:1, 1:0}).astype(np.int64)
    

    
    df_train, df_test = train_test_split(df, test_size=0.2)
    
    # modeling
    losses_dict = {}
    losses_dict["losses"] = []
    losses_dict["auroc"] = []
    losses_dict["auprc"] = []
    
    
    
    losses = []
    _auroc = []
    _auprc = []
    
    random.seed(rand_seed_np)
    np.random.seed(rand_seed_np)
    torch.manual_seed(rand_seed_torch)
    torch.cuda.manual_seed(rand_seed_torch)
    
    
    X_train = df_train["text"]
    y_train = df_train[['gender', 'source']]

    X_test = df_test["text"]
    y_test = df_test[['gender', 'source']]
    
    model = MultiLabel(**model_config)

    model.load_pretrained()

    model.trainModel(X=X_train, y=y_train, device="cuda:0")
    y_pred, y_prob = model.predict(X=X_test, device="cuda:0")
    #print(y_prob)

    
    loss = torch.nn.BCELoss()
    
    _loss = loss(
                torch.FloatTensor(y_prob.to_numpy()), torch.FloatTensor(y_test.to_numpy())
            )
    
    losses.append(_loss.item())
    _auroc.append(metrics.roc_auc_score(y_true=y_test[['gender', 'source']], y_score=y_prob))
    _auprc.append(
                metrics.average_precision_score(y_true=y_test[['gender', 'source']], y_score=y_prob)
            )
    

    losses_dict["losses"].append(losses)
    losses_dict["auroc"].append(_auroc)
    losses_dict["auprc"].append(_auprc)

    
    with open(os.path.join(destination, "multi_results.pkl"), "wb") as f:
        pickle.dump(obj=losses_dict, file=f)
    
    
    

if __name__ == "__main__":
    main()