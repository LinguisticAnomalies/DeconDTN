import numpy as np
import pandas as pd


def load_wls():
    df_wls = pd.read_csv("/edata/TRESTLE/testWLS.tsv", sep="\t")

    df_wls_label = pd.read_csv("/edata/TRESTLE/WLS-labels.csv")

    df_wls_merge = df_wls.merge(
        df_wls_label, left_on="file", right_on="idtlkbnk", how="inner"
    )

    # df_wls_merge.rename(columns={"> 1 sd below mean for normals ages 60-79 (Tombaugh, Kozak, & Rees, 1999) -- normal cutoff = 12+ for 9-12 yrs eductation, 14+ for 13-21 yrs education":
    #                              "label",

    #                             },
    #                     inplace=True
    #                    )

    # df_wls_merge.loc[df_wls_merge['label'] == 'y','label'] = 'Y'

    # condlist = [
    #     df_wls_merge['label'] == 'Y',
    #     df_wls_merge['label'] == 'N',
    #     df_wls_merge['label'].isna()
    # ]
    # choicelist = [
    #     1,
    #     0,
    #     np.nan
    # ]

    # df_wls_merge['label'] = np.select(condlist, choicelist)

    wls_Conditions = [
        (df_wls_merge["age 2011"] <= 79)
        & (df_wls_merge["education"] <= 12)
        & (df_wls_merge["category fluency, scored words named, 2011"] < 12),

        (df_wls_merge["age 2011"] <= 79)
        & (df_wls_merge["education"] > 12)
        & (df_wls_merge["category fluency, scored words named, 2011"] < 14),

        (df_wls_merge["age 2011"] > 79)
        & (df_wls_merge["education"] <= 12)
        & (df_wls_merge["category fluency, scored words named, 2011"] < 10.5),

        (df_wls_merge["age 2011"] > 79)
        & (df_wls_merge["education"] > 12)
        & (df_wls_merge["category fluency, scored words named, 2011"] < 12),
    ]
    wls_Categories = [1, 1, 1, 1]

    df_wls_merge["label"] = np.select(wls_Conditions, wls_Categories, default=0)

    df_wls_merge = df_wls_merge.loc[df_wls_merge["label"].notna(), :].reset_index(
        drop=True
    )

    return df_wls_merge


def load_adress():
    # df_adress_train = pd.read_csv("/edata/ADReSS-IS2020-data/dataframes/adre_train.csv")

    # df_adress_test = pd.read_csv("/edata/ADReSS-IS2020-data/dataframes/adre_test.csv")

    # df_adress = pd.concat([df_adress_train, df_adress_test], ignore_index=True)

    # df_adress.rename(columns={"sentence": "text"}, inplace=True)

    df_adress_dementia = pd.read_csv(
        "/edata/TRESTLE/harmonized-toolkit/transcripts/processed_pitt_dementia.tsv",
        sep="\t",
    )

    df_adress_control = pd.read_csv(
        "/edata/TRESTLE/harmonized-toolkit/transcripts/processed_pitt_control.tsv",
        sep="\t",
    )

    df_adress_dementia["label"] = 1
    df_adress_control["label"] = 0

    df_adress = pd.concat([df_adress_dementia, df_adress_control]).reset_index(
        drop=True
    )

    return df_adress
