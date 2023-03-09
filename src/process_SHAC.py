import itertools
import numpy as np
import pandas as pd
import os
import sys
import glob


def getEventTriggerStatusCode(event_line):
    _ = event_line.split("\t")[1].split(" ")

    ret = {}
    ret["Event"] = _[0].split(":")[0]

    for s in _:
        if s.startswith("Status"):
            ret["StatusCode"] = s.split(":")[1]
            break
        else:
            ret["StatusCode"] = ""

    return ret


def translateStatusCode(attribute, statusCode):

    for a in attribute:
        if a.split(" ")[1] == statusCode:
            codeName = a.split(" ")[-1]
            break
        else:
            codeName = ""

    return codeName


def findEventAttribute(annotation_text):
    _T = []
    _E = []
    _A = []

    events = ["Drug", "Alcohol", "Tobacco"]

    ret = {}

    for _ in annotation_text:
        if _.startswith("T"):
            _T.append(_.split("\t")[0])
        elif _.startswith("E"):
            _E.append(_)
        elif _.startswith("A"):
            _A.append(_)

    for e in events:
        ret[e] = []

        for _event_line in _E:
            etsc = getEventTriggerStatusCode(event_line=_event_line)
            if etsc["Event"] == e:
                status = translateStatusCode(
                    attribute=_A, statusCode=etsc["StatusCode"]
                )
                if status != "":
                    ret[e].append(status)

    return ret


def runEventRetriever(base_dir, file_group):
    ret = {}

    annotation_files = glob.glob(os.path.join(base_dir, file_group, "*.ann"))

    for _file in annotation_files:
        file_id = os.path.basename(_file.split(".")[0])

        with open(_file) as f:
            tmp = f.read().splitlines()

        ret[file_id] = findEventAttribute(annotation_text=tmp)

    return ret


def getBinaryLabel(pt_event_attribute):
    ret = {}
    for k, v in pt_event_attribute.items():
        _ret_event = {}
        for ek, ev in v.items():
            attributes = set(ev)
            if len(attributes) == 0:
                binarylabel = np.nan
            elif ("current" in attributes) or ("past" in attributes):
                binarylabel = True
            else:
                binarylabel = False

            _ret_event[ek] = binarylabel

        ret[k] = _ret_event

    return ret


def getPatientLabel(base_dir, file_group):
    events = ["Drug", "Alcohol", "Tobacco"]
    pt_event_attribute = runEventRetriever(base_dir=base_dir, file_group=file_group)
    pt_event_BLabel = getBinaryLabel(pt_event_attribute=pt_event_attribute)

    df = pd.DataFrame(pt_event_BLabel).T.reset_index().rename(columns={"index": "id"})

    df["SubstanceAgg"] = df[events].sum(axis=1).astype(bool)

    return df


def load_SHAC(base_dir = "/edata/xiruod/SocialDeterminants_SHAC_n2c2_2022/n2c2_sdoh_challenge"):
    # base_dir = "/edata/xiruod/SocialDeterminants_SHAC_n2c2_2022/n2c2_sdoh_challenge"

    # label reader
    ls_labels = []

    tmp = {}
    for c in itertools.product(["train", "dev", "test"], ["uw", "mimic"]):
        file_group = c[0] + "/" + c[1]

        _ = getPatientLabel(base_dir=base_dir, file_group=file_group)
        _["set"] = c[0]
        _["location"] = c[1]
        ls_labels.append(_)

    df_labels = pd.concat(ls_labels).reset_index(drop=True)

    # txt reader
    txt_files = []
    for _ in glob.glob(os.path.join(base_dir, "*/*/*.txt")):
        txt_files.append(_)

    list_txt = []
    for _file in txt_files:
        with open(_file) as f:
            list_txt.append(f.read().replace("\n", " "))

    df_txt = pd.DataFrame(
        {"id": [os.path.basename(i).split(".")[0] for i in txt_files], "text": list_txt}
    )

    # merge annotation and text
    df_all = df_labels.merge(df_txt, on="id", how="inner")

    return df_all


def load_process_SHAC(replaceNA, base_dir="/edata/xiruod/SocialDeterminants_SHAC_n2c2_2022/n2c2_sdoh_challenge"):
    df = load_SHAC(base_dir=base_dir)

    assert replaceNA is not None

    if replaceNA == "all":
        df = df.replace(
            np.NaN, False
        )  # replace NaN in status with False. NOTE: this is an assumption!
    else:
        sys.exit(f"Unsupported replaceNA value: {replaceNA}")

    return df