#!/bin/bash

# python3 runs_xiruo/step104_runExp_backdoorBERT_SHAC.py with \
#     proj_name="exp_backdoorBERT_SHAC_n500_05_02_v1" \
#     pretrained="bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[500] \
#     alpha_test=[0,5,0.1] \
#     v=1


python3 runs_xiruo/step104_runExp_backdoorBERT_SHAC.py with \
    proj_name="exp_backdoorBERT_SHAC_n500_05_02_v10" \
    pretrained="bert-base-uncased" \
    p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[500] \
    alpha_test=[0,5,0.1] \
    v=10
# python3 runs_xiruo/step104_runExp_backdoorBERT_SHAC.py with \
#     proj_name="exp_backdoorBERT_SHAC_05_02" \
#     pretrained="bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[200] \
#     alpha_test=[0,1,0.5]