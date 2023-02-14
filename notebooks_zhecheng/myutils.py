from sklearn.model_selection import train_test_split
from sklearn import metrics
from warnings import warn
from tqdm.notebook import tqdm, trange
import torch
import numpy as np
import pandas as pd



def training_with_CS(DNmodel, settings_list, df, target, lossf, average, sample, config):
    """
    Model training with confounding shift
    args:
        DNModel: the model class needs to implement load_pretrained(), trainModel(), predict()
        setting_list: List of probability combinations, output from number_split()
        df1: data of z1
        df0: data of z0
        target: string, target variable
        lossf: loss function for prediction
        average: average method for multi-class classification, default is 'macro'
    
    """
    losses_dict = {}

    losses_dict['combination'] = []
    losses_dict['losses'] = []
    losses_dict['auroc'] = []
    losses_dict['auprc'] = []
    losses_dict['acc'] = []

    for i, st in enumerate(tqdm(settings_list, desc = "Training different combinations")):
    
        dfs = create_mix(df1 = df[1], df0 = df[0], target=target, setting= st, sample = sample)
    
        if dfs is not None:

            losses_dict['combination'].append(st['mix_param_dict'])

            losses = []
            _auroc = []
            _auprc = []    
            _acc = []

            for i in range(5):

                df_train = dfs['train']

                df_test = dfs['test']


                X_train = df_train["text"]
                y_train = df_train[[target]]

                X_test = df_test["text"]
                y_test = df_test[[target]]

                model = DNmodel(**config)

                model.load_pretrained()

                model.trainModel(X=X_train, y=y_train, device="cuda:0")
                y_pred, y_prob = model.predict(X=X_test, device="cuda:0")



                _loss = lossf()(
                    torch.tensor(y_prob.values),
                    torch.tensor(y_test.values).squeeze(1),
                )

                losses.append(_loss.item())


                _auroc.append(
                    metrics.roc_auc_score(
                        y_true=y_test.values,
                        y_score=y_prob.max(axis = 1), average=average
                    )
                )
                _auprc.append(
                    metrics.average_precision_score(
                        y_true=y_test.values,
                        y_score=y_prob.max(axis = 1), average=average
                    )
                )

                _acc.append(
                    metrics.accuracy_score(
                        y_true = y_test.values,
                        y_pred = y_pred
                    )
                )

            losses_dict['losses'].append(losses)
            losses_dict['auroc'].append(_auroc)
            losses_dict['auprc'].append(_auprc)
            losses_dict['acc'].append(_acc)


    dict_list={}

    for d in losses_dict['combination']:
        for k in d:
            if dict_list.get(k):
                dict_list[k].append(d[k])
            else:
                dict_list[k] = []
                dict_list[k].append(d[k])

    losses_dict.update(dict_list)
    del losses_dict['combination']
    
    return losses_dict



def create_mix(df1, df0, target, setting, sample = False, seed = 2023):
    """ Create a mixture dataset from two source based on pre-set constraints"""
    n_total = len(df1) + len(df0)
    
    # check if there is enough positive samples in each dataset
    n_z0_pos = setting['n_z0_pos_train'] + setting['n_z0_pos_test']
    n_z1_pos = setting['n_z1_pos_train'] + setting['n_z1_pos_test']
    n_z0_neg = setting['n_z0_neg_train'] + setting['n_z0_neg_test']
    n_z1_neg = setting['n_z1_neg_train'] + setting['n_z1_neg_test']
    
    df0_pos = df0[df0[target] == 1]
    df1_pos = df1[df1[target] == 1]
    
    
    df0_neg = df0[df0[target] == 0]
    df1_neg = df1[df1[target] == 0]
    
     
    # for z0 positive    
    if n_z0_pos <= len(df0_pos):        
        df0_train_pos, df0_test_pos = train_test_split(df0_pos, 
                                                       train_size=setting['n_z0_pos_train'], 
                                                       test_size=setting['n_z0_pos_test'], 
                                                       shuffle = True, random_state=seed)
    elif sample: 
        df0_pos_extra = df0_pos.sample(n = n_z0_pos - len(df0_pos), replace = True)
        df0_pos_sampled = pd.concat([df0_pos,df0_pos_extra], axis = 0, ignore_index=True)
        df0_train_pos, df0_test_pos = train_test_split(df0_pos_sampled, 
                                                       train_size=setting['n_z0_pos_train'], 
                                                       test_size=setting['n_z0_pos_test'], 
                                                       shuffle = True, random_state=seed)
    else:
        warn("Set sample equals to True or augment current dataset.")
        return
        
    # for z0 negative
    if n_z0_neg <= len(df0_neg):        
        df0_train_neg, df0_test_neg = train_test_split(df0_neg, 
                                                       train_size=setting['n_z0_neg_train'], 
                                                       test_size=setting['n_z0_neg_test'], 
                                                       shuffle = True, random_state=seed)
    elif sample: 
        df0_neg_extra = df0_neg.sample(n = n_z0_neg - len(df0_neg), replace = True)
        df0_neg_sampled = pd.concat([df0_neg,df0_neg_extra], axis = 0, ignore_index=True)
        df0_train_neg, df0_test_neg = train_test_split(df0_neg_sampled, 
                                                       train_size=setting['n_z0_neg_train'], 
                                                       test_size=setting['n_z0_neg_test'], 
                                                       shuffle = True, random_state=seed)
    else:
        warn("Set sample equals to True or augment current dataset.")
        return
        
    
    
    # for z1 positive    
    if n_z1_pos <= len(df1_pos):        
        df1_train_pos, df1_test_pos = train_test_split(df1_pos, 
                                                       train_size=setting['n_z1_pos_train'], 
                                                       test_size=setting['n_z1_pos_test'], 
                                                       shuffle = True, random_state=seed)
    elif sample: 
        df1_pos_extra = df1_pos.sample(n = n_z1_pos - len(df1_pos), replace = True)
        df1_pos_sampled = pd.concat([df1_pos,df1_pos_extra], axis = 0, ignore_index=True)
        df1_train_pos, df1_test_pos = train_test_split(df1_pos_sampled, 
                                                       train_size=setting['n_z1_pos_train'], 
                                                       test_size=setting['n_z1_pos_test'], 
                                                       shuffle = True, random_state=seed)
    else:
        warn("Set sample equals to True or augment current dataset.")
        return
    
     # for z1 negative
    if n_z1_neg <= len(df1_neg):        
        df1_train_neg, df1_test_neg = train_test_split(df1_neg, 
                                                       train_size=setting['n_z1_neg_train'], 
                                                       test_size=setting['n_z1_neg_test'], 
                                                       shuffle = True, random_state=seed)
    elif sample: 
        df1_neg_extra = df1_neg.sample(n = n_z1_neg - len(df1_neg), replace = True)
        df1_neg_sampled = pd.concat([df1_neg,df1_neg_extra], axis = 0, ignore_index=True)
        df1_train_neg, df1_test_neg = train_test_split(df1_neg_sampled, 
                                                       train_size=setting['n_z1_neg_train'], 
                                                       test_size=setting['n_z1_neg_test'], 
                                                       shuffle = True, random_state=seed)
    else:
        warn("Set sample equals to True or augment current dataset.")
        return
    
    
    # assemble mixed train and test
    df_train = pd.concat([df0_train_pos, df0_train_neg, df1_train_pos, df1_train_neg], axis = 0, ignore_index=True)
    df_test = pd.concat([df0_test_pos, df0_test_neg, df1_test_pos, df1_test_neg], axis = 0, ignore_index=True)
    
    
    return {'train':df_train, 'test':df_test}



def number_split(p_pos_train_z1,
    p_pos_train_z0,
    p_mix_z1,
    alpha_test,
    train_test_ratio=5,
    n_test = 100 # set the number for tests
      ):
    """Get required number of samples for each category"""
    assert isinstance(train_test_ratio, int)
    
    mix_param_dict = confoundSplit(
        p_pos_train_z0=p_pos_train_z0,
        p_pos_train_z1=p_pos_train_z1,
        p_mix_z1=p_mix_z1,
        alpha_test=alpha_test,
    )
    
    if all(0 < mix_param_dict[key] < 1 for key in mix_param_dict.keys() if key not in ['alpha_test','alpha_train']): # assert all probability between 0 and 1
        
        n_train = n_test * train_test_ratio

        n_z1_train = round(n_train * mix_param_dict["C_z"])
        n_z1_test = round(n_test * mix_param_dict["C_z"])

        n_z0_train = n_train - n_z1_train
        n_z0_test = n_test - n_z1_test

        n_z1_p_train = round(n_z1_train * mix_param_dict["p_pos_train_z1"])
        n_z0_p_train = round(n_z0_train * mix_param_dict["p_pos_train_z0"])

        n_z1_p_test = round(n_z1_test * mix_param_dict['p_pos_test_z1'])
        n_z0_p_test = round(n_z0_test * mix_param_dict['p_pos_test_z0'])


        n_z1_n_train = n_z1_train - n_z1_p_train
        n_z0_n_train = n_z0_train - n_z0_p_train

        n_z1_n_test = n_z1_test - n_z1_p_test
        n_z0_n_test = n_z0_test - n_z0_p_test

        ans = {
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_z0_pos_train": n_z0_p_train,
                    "n_z0_neg_train": n_z0_n_train,
                    "n_z0_pos_test": n_z0_p_test,
                    "n_z0_neg_test": n_z0_n_test,
                    "n_z1_pos_train": n_z1_p_train,
                    "n_z1_neg_train": n_z1_n_train,
                    "n_z1_pos_test": n_z1_p_test,
                    "n_z1_neg_test": n_z1_n_test,
                    "mix_param_dict": mix_param_dict
                }
        
        if all(ans[key] > 0 for key in ans.keys() if key != 'mix_param_dict'):
            
            return ans
               
        else:
            print("Invalid sample numbers ", "n_z1_neg_train:", ans["n_z1_neg_train"], 
                     "n_z0_neg_train:", ans["n_z0_neg_train"], 
                     "n_z1_neg_test:", ans["n_z1_neg_test"], 
                     "n_z0_neg_test:", ans["n_z0_neg_test"])
            return None
    
    else:
        #print(mix_param_dict)
        print(f"Invalid test set probability P(Y=1|Z=0):{mix_param_dict['p_pos_test_z0']}, P(Y=1|Z=1):{mix_param_dict['p_pos_test_z1']}")
    
    return None    



def confoundSplit(p_pos_train_z1, p_pos_train_z0, p_mix_z1, alpha_test):
    """Calculate probability constraint given some priors"""

    assert 0 <= p_pos_train_z1 <= 1
    assert 0 <= p_pos_train_z0 <= 1
    assert 0 <= p_mix_z1 <= 1
    assert alpha_test >= 0

    C_z = p_mix_z1

    p_mix_z0 = 1 - p_mix_z1

    # C_y = p_train(y=1) = p_train(z=0) * p_train(y=1|z=0) + p_train(z=1) * p_train(y=1|z=1)
    # C_y = p_test(y=1) = p_test(z=0) * p_test(y=1|z=0) + p_test(z=1) * p_test(y=1|z=1)
    C_y = p_mix_z0 * p_pos_train_z0 + p_mix_z1 * p_pos_train_z1

    p_pos_test_z0 = C_y / (1 - (1 - alpha_test) * C_z)
    p_pos_test_z1 = alpha_test * p_pos_test_z0

    # alpha_test = p_pos_test_z1 / p_pos_test_z0
    alpha_train = p_pos_train_z1 / p_pos_train_z0

    return {
        "p_pos_train_z0": p_pos_train_z0,
        "p_pos_train_z1": p_pos_train_z1,
        "p_pos_train": C_y,
        "p_pos_test": C_y,
        "p_mix_z0": p_mix_z0,
        "p_mix_z1": p_mix_z1,
        "alpha_train": alpha_train,
        "alpha_test": alpha_test,
        "p_pos_test_z0": p_pos_test_z0,
        "p_pos_test_z1": p_pos_test_z1,
        "C_y": C_y,
        "C_z": C_z,
    }
