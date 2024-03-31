import pandas as pd
import pathlib


def load_HateSpeech_dynGen(
    file="/bime-munin/xiruod/data/hateSpeech_Bulla2023/Dynamically-Generated-Hate-Speech-Dataset/Dynamically Generated Hate Dataset v0.2.3.csv",
):
    # (1) dynGen
    df_dynGen = pd.read_csv(
        file,
    )

    df_dynGen["label"] = df_dynGen["label"].map({"hate": "hate", "nothate": "nothate"})
    df_dynGen["dfSource"] = "dynGen"
    df_dynGen["label_binary"] = df_dynGen["label"].map({"hate": 1, "nothate": 0})
    return df_dynGen


def load_HateSpeech_wsf(
    file="/bime-munin/xiruod/data/hateSpeech_Bulla2023/hate-speech-dataset/all_files/",
    annotation_file="/bime-munin/xiruod/data/hateSpeech_Bulla2023/hate-speech-dataset/annotations_metadata.csv",
):
    # (2)  wsf
    ls_allFiles = pathlib.Path(file).glob("*.txt")

    ls_id = []
    ls_text = []

    for ifile in ls_allFiles:
        ls_id.append(ifile.name.split(".txt")[0])
        with open(ifile, "r") as f:
            ls_text.append(f.read())

    df_wsf_raw = pd.DataFrame({"file_id": ls_id, "text": ls_text})

    df_wsf_annotation = pd.read_csv(annotation_file)

    df_wsf = df_wsf_raw.merge(df_wsf_annotation, on="file_id", how="inner")

    df_wsf = df_wsf[df_wsf["label"].isin(["hate", "noHate"])].reset_index(drop=True)

    df_wsf["label"] = df_wsf["label"].map({"hate": "hate", "noHate": "nothate"})
    df_wsf["dfSource"] = "wsf"
    df_wsf["label_binary"] = df_wsf["label"].map({"hate": 1, "nothate": 0})

    return df_wsf


def load_HateSpeech_youtube(
    file="/bime-munin/xiruod/data/hateSpeech_Bulla2023/HS_YouTube/IMSyPP_EN_YouTube_comments_evaluation_context.csv",
):
    df_yt = pd.read_csv(file)
    df_yt = df_yt.query("Type.isin(['0. appropriate', '2. offensive'])").reset_index(
        drop=True
    )

    df_yt["label_binary"] = df_yt["Type"].map({"2. offensive": 1, "0. appropriate": 0})
    df_yt = df_yt.rename(columns={"Text": "text"})

    return df_yt
