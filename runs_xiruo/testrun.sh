#!/bin/bash

python3 runs_xiruo/step104_runExp_backdoorBERT_SHAC.py with \
    proj_name="exp_backdoorBERT_SHAC_05_02TEST" \
    pretrained="bert-base-uncased" \
    p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[150]