import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import warnings


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


def confoundSplitNumbers(
    df0,
    df1,
    df0_label,
    df1_label,
    p_pos_train_z1,
    p_pos_train_z0,
    p_mix_z1,
    alpha_test,
    train_test_ratio=4,
    n_test=None,  # set the number for tests
    n_test_error = 0,
):

    """
    df0_label, df1_label: 0/1, or True/False coded


    """
    assert df0[df0_label].isin([0, 1]).all(axis=0)
    assert df1[df1_label].isin([0, 1]).all(axis=0)

    mix_param_dict = confoundSplit(
        p_pos_train_z0=p_pos_train_z0,
        p_pos_train_z1=p_pos_train_z1,
        p_mix_z1=p_mix_z1,
        alpha_test=alpha_test,
    )

    N_df0_pos = (df0[df0_label] == 1).sum()
    N_df0_neg = (df0[df0_label] == 0).sum()

    N_df1_pos = (df1[df1_label] == 1).sum()
    N_df1_neg = (df1[df1_label] == 0).sum()

    N_df0 = N_df0_pos + N_df0_neg
    N_df1 = N_df1_pos + N_df1_neg

    n_df0_test_pos = math.floor(N_df0 / (train_test_ratio + 1))

    while n_df0_test_pos > 0:

        n_df0_test_neg = math.floor(
            n_df0_test_pos
            / mix_param_dict["p_pos_test_z0"]
            * (1 - mix_param_dict["p_pos_test_z0"])
        )

        n_df0_train_pos = math.floor(
            (n_df0_test_pos + n_df0_test_neg)
            * train_test_ratio
            * mix_param_dict["p_pos_train_z0"]
        )
        n_df0_train_neg = math.floor(
            (n_df0_test_pos + n_df0_test_neg)
            * train_test_ratio
            * (1 - mix_param_dict["p_pos_train_z0"])
        )

        n_df1_train = math.floor(
            mix_param_dict["C_z"]
            / (1 - mix_param_dict["C_z"])
            * (n_df0_train_pos + n_df0_train_neg)
        )
        n_df1_train_pos = math.floor(n_df1_train * mix_param_dict["p_pos_train_z1"])
        n_df1_train_neg = math.floor(
            n_df1_train * (1 - mix_param_dict["p_pos_train_z1"])
        )

        n_df1_test = math.floor(n_df1_train / train_test_ratio)
        n_df1_test_pos = math.floor(n_df1_test * mix_param_dict["p_pos_test_z1"])
        n_df1_test_neg = math.floor(n_df1_test * (1 - mix_param_dict["p_pos_test_z1"]))

        test1 = 0 < (n_df0_train_pos + n_df0_test_pos) <= N_df0_pos
        test2 = 0 < (n_df0_train_neg + n_df0_test_neg) <= N_df0_neg

        test3 = 0 < (n_df1_train_pos + n_df1_test_pos) <= N_df1_pos
        test4 = 0 < (n_df1_train_neg + n_df1_test_neg) <= N_df1_neg

        test5 = 0 < n_df0_train_pos
        test6 = 0 < n_df0_train_neg
        test7 = 0 < n_df1_train_pos
        test8 = 0 < n_df1_train_neg

        test9 = 0 < n_df0_test_pos
        test10 = 0 < n_df0_test_neg
        test11 = 0 < n_df1_test_pos
        test12 = 0 < n_df1_test_neg

        tester_positive_number = (
            test1
            and test2
            and test3
            and test4
            and test5
            and test6
            and test7
            and test8
            and test9
            and test10
            and test11
            and test12
        )

        tester_n_test = (
            n_df0_test_pos + n_df0_test_neg + n_df1_test_pos + n_df1_test_neg
        )


        if tester_positive_number:

            ret = {
                "n_df0_train_pos": n_df0_train_pos,
                "n_df0_test_pos": n_df0_test_pos,
                "n_df0_train_neg": n_df0_train_neg,
                "n_df0_test_neg": n_df0_test_neg,
                "n_df1_train_pos": n_df1_train_pos,
                "n_df1_test_pos": n_df1_test_pos,
                "n_df1_train_neg": n_df1_train_neg,
                "n_df1_test_neg": n_df1_test_neg,
                "mix_param_dict": mix_param_dict,
            }
            if n_test is None:
                return ret
            else:
                if (n_test - n_test_error) <= tester_n_test <= (n_test + n_test_error):
                    return ret
                else:
                    n_df0_test_pos -= 1
        else:
            n_df0_test_pos -= 1

        if n_df0_test_pos == 0:
            return None


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
        warnings.warn("Set sample equals to True or augment current dataset.")
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
        warnings.warn("Set sample equals to True or augment current dataset.")
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
        warnings.warn("Set sample equals to True or augment current dataset.")
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
        warnings.warn("Set sample equals to True or augment current dataset.")
        return


    # assemble mixed train and test
    df_train = pd.concat([df0_train_pos, df0_train_neg, df1_train_pos, df1_train_neg], axis = 0, ignore_index=True)
    df_test = pd.concat([df0_test_pos, df0_test_neg, df1_test_pos, df1_test_neg], axis = 0, ignore_index=True)


    return {'train':df_train, 'test':df_test, 'setting': setting}



def number_split(p_pos_train_z1,
    p_pos_train_z0,
    p_mix_z1,
    alpha_test,
    train_test_ratio=5,
    n_test = 100, # set the number for tests
    verbose = True
      ):
    """Get required number of samples for each category"""
    assert isinstance(train_test_ratio, int)
    assert isinstance(n_test, int)
    
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

        elif verbose:
            print("Invalid sample numbers ", [(key, val) for key, val in ans.items() if key != 'mix_param_dict'])
            return None

    elif verbose:
        print(f"Invalid test set probability P(Y=1|Z=0):{mix_param_dict['p_pos_test_z0']}, P(Y=1|Z=1):{mix_param_dict['p_pos_test_z1']}")

    return None



def confoundSplitDF(
    df0,
    df1,
    df0_label,
    df1_label,
    p_pos_train_z1,
    p_pos_train_z0,
    p_mix_z1,
    alpha_test,
    train_test_ratio=4,
    random_state=186,
    n_test = None,  # set the number for tests
    n_test_error=0,  # set the error range for the number of tests
):

    df0 = df0.copy()
    df1 = df1.copy()

    ret = confoundSplitNumbers(
        df0=df0,
        df1=df1,
        df0_label=df0_label,
        df1_label=df1_label,
        p_pos_train_z1=p_pos_train_z1,
        p_pos_train_z0=p_pos_train_z0,
        p_mix_z1=p_mix_z1,
        alpha_test=alpha_test,
        train_test_ratio=train_test_ratio,
        n_test=n_test,
        n_test_error=n_test_error,
    )

    def SampleDrop(df, label_name, label, n):
        ret_df = df[df[label_name] == label].sample(n, random_state=random_state)
        df.drop(ret_df.index, inplace=True)

        return ret_df

    sample_df0_train_pos = SampleDrop(
        df=df0, label_name=df0_label, label=1, n=ret["n_df0_train_pos"]
    )
    sample_df0_train_neg = SampleDrop(
        df=df0, label_name=df0_label, label=0, n=ret["n_df0_train_neg"]
    )
    sample_df0_test_pos = SampleDrop(
        df=df0, label_name=df0_label, label=1, n=ret["n_df0_test_pos"]
    )
    sample_df0_test_neg = SampleDrop(
        df=df0, label_name=df0_label, label=0, n=ret["n_df0_test_neg"]
    )

    sample_df1_train_pos = SampleDrop(
        df=df1, label_name=df1_label, label=1, n=ret["n_df1_train_pos"]
    )
    sample_df1_train_neg = SampleDrop(
        df=df1, label_name=df1_label, label=0, n=ret["n_df1_train_neg"]
    )
    sample_df1_test_pos = SampleDrop(
        df=df1, label_name=df1_label, label=1, n=ret["n_df1_test_pos"]
    )
    sample_df1_test_neg = SampleDrop(
        df=df1, label_name=df1_label, label=0, n=ret["n_df1_test_neg"]
    )

    sample_df0_train = (
        pd.concat([sample_df0_train_pos, sample_df0_train_neg], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    sample_df0_test = (
        pd.concat([sample_df0_test_pos, sample_df0_test_neg], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    sample_df1_train = (
        pd.concat([sample_df1_train_pos, sample_df1_train_neg], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    sample_df1_test = (
        pd.concat([sample_df1_test_pos, sample_df1_test_neg], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return {
        "sample_df0_train": sample_df0_train,
        "sample_df0_test": sample_df0_test,
        "sample_df1_train": sample_df1_train,
        "sample_df1_test": sample_df1_test,
        "stats": ret,
    }



def confoundSplitDFMultiLevel(df, 
                              z_Categories, y_Categories, 
                              z_column, y_column,
                              p_train_y_given_z, p_test_y_given_z, p_z, 
                              n_test=100, n_error=0,
                              train_test_ratio=4, seed=2671):
    """This

    Args:
        df (pd.DataFrame): DataFrame, containing ALL data
        z_Categories (list): unique list for confounder (z) values
        y_Categories (list): unique list for outcome (y) values
        z_column (str): name of confounder column
        y_column (str): name of outcome column
        p_train_y_given_z (list or 2D-array): P_train(Y|Z): ordered as 2D-array/nested list, where each row represents one event of Z. NOTE: row sum == 1
        p_test_y_given_z (list or 2D-array): P_test(Y|Z). Same as p_train_y_given_z
        p_z (list or 1D-array): distribution of confounding variable z
        n_test (int, optional): number of examples in the testing set. Defaults to 100.
        n_error (int, optional): number for error (+/-) when there is no exact matching for n_test. Defaults to 0.
        train_test_ratio (int, optional): train:test number ratio. Defaults to 4.
        seed (int, optional): random seed for train_test_split. Defaults to 2671.

    Returns:
        df_collect: list of dictionaries, each of which is for one combination of y and z. For each:
            {"df_train":_df_train, 
                "df_test":_df_test, 
                "y":_df[y_column].unique().tolist(), 
                "z":_df[z_column].unique().tolist()
            }

    """
    

    # convert to np.ndarray
    if isinstance(p_train_y_given_z, list):
        p_train_y_given_z = np.array(p_train_y_given_z)

    if isinstance(p_test_y_given_z, list):
        p_test_y_given_z = np.array(p_test_y_given_z)

    if isinstance(p_z, list):
        p_z = np.array(p_z)
    
    
    # quality check
    assert np.all(p_train_y_given_z.sum(axis=1) == 1)
    assert np.all(p_test_y_given_z.sum(axis=1) == 1)
    assert np.all(p_z.sum() == 1)

    assert p_train_y_given_z.shape == p_test_y_given_z.shape

    # get number of categories for Y and Z
    n_zC = len(z_Categories)
    n_yC = len(y_Categories)

    assert p_train_y_given_z.shape == (n_zC, n_yC)
    
    
    
    # calculate number for training and testing sets given probabilities
    n_train = n_test * train_test_ratio  # train:test ratio
    
    n_train_y_given_z = (p_train_y_given_z * (p_z * n_train).round(0).repeat(n_yC).reshape(n_zC, n_yC)).round(0)
    n_test_y_given_z  = (p_test_y_given_z  * (p_z * n_test ).round(0).repeat(n_yC).reshape(n_zC, n_yC)).round(0)
    
    TESTER_n_test = (n_test - n_error) <= n_test_y_given_z.sum() <= (n_test + n_error)
    if not TESTER_n_test:
        return None
    
    
    # calculate how many examples the original data have, for each z,y combination
    # and create sub-df's for each combination
    full_df_list = []
    full_shape_ls = []
    for iz, iy in itertools.product(z_Categories, y_Categories):
        _df = df[(df[z_column] == iz) & (df[y_column] == iy)]

        full_shape_ls.append(len(_df))
        full_df_list.append(_df)

    TESTER_full_shape = np.all((n_test_y_given_z + n_train_y_given_z) <= np.array(full_shape_ls).reshape(n_zC, n_yC))
    if not TESTER_full_shape:
        return None
    
    
    # iterate through every sub-df, get train-test split
    df_collect = []
    for idx, _df in enumerate(full_df_list):
        n_needed_train = int(n_train_y_given_z.flatten()[idx])
        n_needed_test =  int(n_test_y_given_z.flatten()[idx])

        if (n_needed_test == 0) or (n_needed_train == 0):
            _df_train = _df_test = None
        else:

            _df_train, _df_test = train_test_split(_df,
                                                   train_size=n_needed_train,
                                                   test_size=n_needed_test,
                                                   shuffle=True, random_state=seed
                                                  )

        _ret = {"df_train":_df_train, 
                "df_test":_df_test, 
                "y":_df[y_column].unique().tolist(), 
                "z":_df[z_column].unique().tolist()
               }

        df_collect.append(_ret)

    return df_collect
    