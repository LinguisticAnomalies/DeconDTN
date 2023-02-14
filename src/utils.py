import math
import pandas as pd


def confoundSplit(p_pos_train_z1, p_pos_train_z0, p_mix_z1, alpha_test):

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
    
    # Initial a start point
    # N_test(Y = 1|Z=0) = N_train(Y = 1|Z=0) / 5
    n_df0_test_pos = math.floor(N_df0_pos / (train_test_ratio + 1))

    while n_df0_test_pos > 0:
        
        # N_test(Y = 0, Z = 0) = N_test(Y = 1, Z = 0) * (P_test(Y = 0| Z = 0) / P_test(Y = 1| Z = 0))
        n_df0_test_neg = math.floor(
            n_df0_test_pos
            / mix_param_dict["p_pos_test_z0"]
            * (1 - mix_param_dict["p_pos_test_z0"])
        )
        # N_train(Y = 0, Z = 0) = N_test(Z = 0) * train_test_ratio * P_train(Y = 0|Z = 0)
        n_df0_train_neg = math.floor(
            (n_df0_test_pos + n_df0_test_neg)
            * train_test_ratio
            * (1 - mix_param_dict["p_pos_train_z0"])
        )
        
        # N_train(Y = 1, Z = 0) = N(Y = 1, Z = 0) - N_test(Y = 1, Z = 0)
        n_df0_train_pos = math.floor((n_df0_test_pos + n_df0_test_neg)  * train_test_ratio * mix_param_dict['p_pos_train_z0'])
        
        # N_test(Y = 1, Z = 1) = N(Y = 1, Z = 0) * alpha * (P(Z=1)/P(Z=0))
        n_df1_test_pos = math.floor(mix_param_dict['alpha_test'] * mix_param_dict['C_z']/(1-mix_param_dict['C_z']) * n_df0_test_pos)
        
        # N_train(Z = 1) = N_train(Z = 0) * P(Z = 1)/P(Z = 0)
        n_df1_train = math.floor(
            mix_param_dict["C_z"]
            / (1 - mix_param_dict["C_z"])
            * (n_df0_train_pos + n_df0_train_neg)
        )
        
        # N_train(Y = 1, Z = 1) = N_train(Z = 1) * P(Y = 1 | Z = 1)
        n_df1_train_pos = math.floor(n_df1_train * mix_param_dict["p_pos_train_z1"])
        
        # N_train(Y = 0, Z = 1) = N_train(Z = 1)  - N_train(Y = 1, Z = 1)
        n_df1_train_neg = n_df1_train  - n_df1_train_pos
        
        n_df1_test = math.floor(n_df1_train / train_test_ratio)
        
        n_df1_test_neg = n_df1_test - n_df1_test_pos

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
                "n_train": n_df0_train_pos + n_df0_train_neg + n_df1_train_pos + n_df1_train_neg,
                "n_test": n_df0_test_pos + n_df0_test_neg + n_df1_test_pos + n_df1_test_neg
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



        # if (
        #     test1
        #     and test2
        #     and test3
        #     and test4
        #     and test5
        #     and test6
        #     and test7
        #     and test8
        #     and test9
        #     and test10
        #     and test11
        #     and test12
        # ):

        #     return {
        #         "n_df0_train_pos": n_df0_train_pos,
        #         "n_df0_test_pos": n_df0_test_pos,
        #         "n_df0_train_neg": n_df0_train_neg,
        #         "n_df0_test_neg": n_df0_test_neg,
        #         "n_df1_train_pos": n_df1_train_pos,
        #         "n_df1_test_pos": n_df1_test_pos,
        #         "n_df1_train_neg": n_df1_train_neg,
        #         "n_df1_test_neg": n_df1_test_neg,
        #         "mix_param_dict": mix_param_dict,
        #     }
        # else:
        #     n_df0_test_pos -= 1

        if n_df0_test_pos == 0:
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

   