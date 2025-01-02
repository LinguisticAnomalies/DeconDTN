from copy import deepcopy
import pandas as pd

import random
from subprocess import getoutput


def runEDA(
    text_inputs: pd.Series,
    file_tmp_for_eda: str,
    file_eda_output: str,
) -> pd.DataFrame:

    num_aug = 4

    eda_input = deepcopy(text_inputs)

    eda_input = eda_input.str.replace("\n", " ")

    eda_input = eda_input.str.replace("\t", " ")

    eda_input.to_csv(file_tmp_for_eda, index=False, sep="\t", header=False)

    cmd_txt_template = """
                    cd /home/NETID/xiruod/projects/eda_GitHub/eda_nlp

                    python code/augment.py \
                        --input={} \
                        --output={} \
                        --num_aug={} \
                        --alpha_sr=0.1 \
                        --alpha_rd=0.1 \
                        --alpha_ri=0.1 \
                        --alpha_rs=0.1

                    """

    random.seed(102)

    cmd_txt = cmd_txt_template.format(file_tmp_for_eda, file_eda_output, num_aug)

    output = getoutput(cmd_txt)

    df_ret = pd.read_csv(file_eda_output, sep="\t", header=None)

    df_ret.columns = ["discard", "text"]

    return df_ret
