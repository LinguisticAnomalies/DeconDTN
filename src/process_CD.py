import sys
import numpy as np
import pandas as pd


sys.path.insert(
    0,
    "/edata/CohenLybarger/xiruo_Project/cognitive_distortions_Transfer_Learning_GitLab/",
)

from src.data_load import load_avh, load_r56


def load_cd():
    df_avh = load_avh(
        distortions=None,
        use_context=False,
        combineLabel=False,
        n_context_previous=None,
    )

    df_r56 = load_r56(distortions=None)

    df_r56.rename(columns={"Text": "text"}, inplace=True)
    df_r56 = df_r56[~df_r56['text'].isna()].reset_index(drop=True)

    df_avh = df_avh[~df_avh['text'].isna()].reset_index(drop=True)

    df_avh["label_binary"] = df_avh["AD"].astype(int)
    df_avh["label"] = df_avh["AD"].astype(int)
    df_avh["dfSource"] = "avh"

    df_r56["label_binary"] = df_r56["AD"].astype(int)
    df_r56["label"] = df_r56["AD"].astype(int)
    df_r56["dfSource"] = "r56"

    return {"avh": df_avh, "r56": df_r56}
