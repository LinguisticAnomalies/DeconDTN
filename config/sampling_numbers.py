import numpy as np

HateSpeech_DICT = {
    "PickC-0": {
        "p_pos_train_z0_ls": [0.3],
        "p_pos_train_z1_ls": [0.3],
        "p_mix_z1_ls": np.arange(0.1, 0.9, 0.1),
    },
    "PickC-1": {
        "p_pos_train_z0_ls": np.arange(0, 1, 0.1),
        "p_pos_train_z1_ls": np.arange(0, 1, 0.1),
        "p_mix_z1_ls": np.arange(0.1, 0.9, 0.1),
    }
}


SHAC_DICT = {
    "PickC-0": {
        "p_pos_train_z0_ls": np.arange(0, 1, 0.1),
        "p_pos_train_z1_ls": np.arange(0, 1, 0.1),
        "p_mix_z1_ls": np.arange(0, 1, 0.05),
    }
}