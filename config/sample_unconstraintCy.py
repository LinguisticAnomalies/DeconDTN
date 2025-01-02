import numpy as np

SAMPLE_CONFIG = {
    "AlphaTrain_1.5-Cz_0.5": {
        "alpha_train": [1.5, 1, 0.6667],
        "p_pos_train_z0_ls": [0.5, 0.3, 2/5,3/5, 6/25, 9/25],
        "p_pos_train_z1_ls": [0.5, 0.3, 2/5,3/5, 6/25, 9/25],
        "C_y": [0.5, 0.3],
        "p_mix_z1_ls": [0.5]
    },
    
    "AlphaTrain_10-Cz_0.5": {
        "alpha_train": [10, 1, 0.1],
        "p_pos_train_z0_ls": [0.5, 0.3, 1/11, 10/11, 3/55, 30/55],
        "p_pos_train_z1_ls": [0.5, 0.3, 1/11, 10/11, 3/55, 30/55],
        "C_y": [0.5, 0.3],
        "p_mix_z1_ls": [0.5]
    },

    "AlphaTrain_5-Cz_0.5": {
        "alpha_train": [5, 1, 0.2],
        "p_pos_train_z0_ls": [0.5, 0.3, 3/30, 15/30, 5/30, 25/30],
        "p_pos_train_z1_ls": [0.5, 0.3, 3/30, 15/30, 5/30, 25/30],
        "C_y": [0.5, 0.3],
        "p_mix_z1_ls": [0.5]
    }
}