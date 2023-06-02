import sys
import os
import random
import argparse

import itertools
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("./"), "..")))
from src.BinaryClassificationNN import BinaryClassificationNN
from src.MultiLabel import MultiLabel

# model configuration
rand_seed = 2023
num_labels = 2
    
pretrained = "bert-base-uncased"
device = "cuda:0"

max_length = 256
num_epochs = 10
hidden_dropout_prob = 0.2
num_warmup_steps = 50
batch_size = 10
    
lr = 1e-5
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

# def split_data(df, target, size = 0.4):
    
#     df_male = df[df['gender'] == target]
#     df_male_train, df_test = train_test_split(df_male, test_size=size, random_state=rand_seed)
#     df_train = df.drop(index = df_test.index)

#     return df_train, df_test


def split_on_id(df, size, rand_seed):
    train_id, test_id = train_test_split(df['id'].unique(), test_size=size, random_state=rand_seed)
    df_train = df[df['id'].isin(train_id)]
    df_test = df[df['id'].isin(test_id)]

    return df_train, df_test

def cross_validation(df, rand_seed, num_folds = 5):
    id = df['id'].unique()
    kf = KFold(n_splits=num_folds, shuffle=True,random_state=rand_seed)
    return kf.split(id)


def main():
    print("Running experiments using 80%/20% split(5 folds)")
    parser = argparse.ArgumentParser(description='Train a binary classification model use bert-base')
    parser.add_argument('-d', '--data', default='pitts_all')
    parser.add_argument('-t', '--target', default='label')
    parser.add_argument('-s', '--subsample', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('-e', '--exp', default='last')
    parser.add_argument('-c', '--clean', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(os.path.join('data', f"processed_{args.data}.csv"))
    target = args.target 

    if args.clean:
        df['text'] = df['text'].str.replace('[^\w\s]','', regex = True)

    if args.subsample:
        df1_m = df[(df['label'] == 1) & (df['gender'] == 'male')]['id'].drop_duplicates()
        df1_f = df[(df['label'] == 1) & (df['gender'] == 'female')]['id'].drop_duplicates()
        df0_m = df[(df['label'] == 0) & (df['gender'] == 'male')]['id'].drop_duplicates()
        df0_f = df[(df['label'] == 0) & (df['gender'] == 'female')]['id'].drop_duplicates()
        #print(len(df1_m), len(df1_f), len(df0_m), len(df0_f))
    
        if len(df1_m) > len(df1_f):
            df1_m = df1_m.sample(n = len(df1_f), random_state = rand_seed)
        else:
            df1_f = df1_f.sample(n = len(df1_m), random_state = rand_seed)
        
        if len(df0_m) > len(df0_f):
            df0_m = df0_m.sample(n = len(df0_f), random_state = rand_seed)
        else:
            df0_f = df0_f.sample(n = len(df0_m), random_state = rand_seed)
        
        df_id = pd.concat([df1_m, df1_f, df0_m, df0_f])
        df = df[df['id'].isin(df_id)]

    #df = df.groupby('id').last().reset_index()
    print(pd.crosstab(df['label'], df['gender']))

    eval_metrics = {"all":{"accuracy": [], "loss": [], "auprc": [], "auroc": [], "f1": []},
                    "male": {"accuracy": [], "loss": [], "auprc": [], "auroc": [], "f1": []},
                    "female": {"accuracy": [], "loss": [], "auprc": [], "auroc": [], "f1": []}}

    # cross validation
    for train_id, test_id in cross_validation(df, rand_seed):
        print(f"Number of training samples: {len(train_id)}, Number of test samples: {len(test_id)}")
        # train_test split
        # train test split on id
        if args.exp == 'all':
            df_train, df_test = df[df['id'].isin(train_id)], df[df['id'].isin(test_id)]
    
    
        elif args.exp == 'last':
            df = df.groupby('id').last().reset_index()
            df_train, df_test = df[df['id'].isin(train_id)], df[df['id'].isin(test_id)]

        
        #df_train, df_test = split_on_id(df, size=0.2,rand_seed=rand_seed)
        #df_test = df_test[df_test['gender'] == args.gender]
        X_train = df_train["text"]
        y_train = df_train[target].to_frame()

    
        
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)
        
        # Model training
        model = BinaryClassificationNN(**model_config)
        model.load_pretrained()
        model.trainModel(X=X_train, y=y_train, device="cuda:0")
        
        
        # Evaluation
        for gender in [['male', 'female'], ['male'], ['female']]:
      
            df_test_sub = df_test[df_test['gender'].isin(gender)]
            X_test = df_test_sub["text"]
            y_test = df_test_sub[target].to_frame()
            if len(gender) == 2:
                gender = ['all']
            print(gender[0])
            y_pred, y_prob = model.predict(X=X_test, device="cuda:0")
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(torch.tensor(y_prob.values), torch.tensor(y_test.values).squeeze(1))
            acc = metrics.accuracy_score(y_true = y_test.values, y_pred=y_pred)
            auroc = metrics.roc_auc_score(y_true=y_test.values,y_score=y_prob.iloc[:,1])
            auprc = metrics.average_precision_score(y_true=y_test.values,y_score=y_prob.iloc[:,1])
            f1 = metrics.f1_score(y_true = y_test.values, y_pred = y_pred)
            #cm = metrics.confusion_matrix(y_true = y_test.values,y_pred = y_pred)
            print(f"Train on {args.exp}, Test on {' and '.join(gender)}")
            print(f"Loss: {loss}, \nAccuracy: {acc}, \nAUROC: {auroc}, \nAUPRC: {auprc}, \nF1: {f1}")
            eval_metrics[gender[0]]['accuracy'].append(acc)
            eval_metrics[gender[0]]['loss'].append(loss)
            eval_metrics[gender[0]]['auroc'].append(auroc)
            eval_metrics[gender[0]]['auprc'].append(auprc)
            eval_metrics[gender[0]]['f1'].append(f1)


        #PATH = f"model/{args.data}-{args.target}.pt"
        
    print("Rsults from 5-fold cross validation:")
    print(f"Average Overall Accuracy: {np.mean(eval_metrics['all']['accuracy'])}, \nAverage F1: {np.mean(eval_metrics['all']['f1'])}, \nAverage Loss: {np.mean(eval_metrics['all']['loss'])}, \nAverage AUROC: {np.mean(eval_metrics['all']['auroc'])}, \nAverage AUPRC: {np.mean(eval_metrics['all']['auprc'])}")
    print('------------------')
    print(f"Average Male Accuracy: {np.mean(eval_metrics['male']['accuracy'])}, \nAverage F1: {np.mean(eval_metrics['male']['f1'])}, \nAverage Loss: {np.mean(eval_metrics['male']['loss'])}, \nAverage AUROC: {np.mean(eval_metrics['male']['auroc'])}, \nAverage AUPRC: {np.mean(eval_metrics['male']['auprc'])}")
    print('------------------')
    print(f"Average Female Accuracy: {np.mean(eval_metrics['female']['accuracy'])}, \nAverage F1: {np.mean(eval_metrics['female']['f1'])}, \nAverage Loss: {np.mean(eval_metrics['female']['loss'])}, \nAverage AUROC: {np.mean(eval_metrics['female']['auroc'])}, \nAverage AUPRC: {np.mean(eval_metrics['female']['auprc'])}")
    print('------------------')




    #torch.save(model, PATH)
    
    # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()
    
    # plt.savefig(f"fig/{args.exp}-{'_'.join(gender)}.png")

    
if __name__ == '__main__':
    main()