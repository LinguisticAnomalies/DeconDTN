#!/bin/bash



# python3 runs_xiruo/step101_runExp.py with proj_name="exp_01_05" p_pos_train_z0=[0.1] p_pos_train_z1=[0.5] n_test=[150]
# python3 runs_xiruo/step101_runExp.py with proj_name="exp_02_04" p_pos_train_z0=[0.2] p_pos_train_z1=[0.4] n_test=[150]
# python3 runs_xiruo/step101_runExp.py with proj_name="exp_03_06" p_pos_train_z0=[0.3] p_pos_train_z1=[0.6] n_test=[150]

# python3 runs_xiruo/step101_runExp.py with proj_name="exp_01_05_N100" p_pos_train_z0=[0.1] p_pos_train_z1=[0.5] p_mix_z1=[0.05,0.99,0.05] n_test=[100] n_valid_high=1
# python3 runs_xiruo/step101_runExp.py with proj_name="exp_02_04_N100" p_pos_train_z0=[0.2] p_pos_train_z1=[0.4] p_mix_z1=[0.05,0.99,0.05] n_test=[100] n_valid_high=1
# python3 runs_xiruo/step101_runExp.py with proj_name="exp_03_06_N100" p_pos_train_z0=[0.3] p_pos_train_z1=[0.6] p_mix_z1=[0.05,0.99,0.05] n_test=[100] n_valid_high=1
# python3 runs_xiruo/step101_runExp.py with proj_name="exp_05_05_N100" p_pos_train_z0=[0.5] p_pos_train_z1=[0.5] p_mix_z1=[0.05,0.99,0.05] n_test=[100] n_valid_high=1




# TODO: re-run?
# python3 runs_xiruo/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_01_05" p_pos_train_z0=[0.1] p_pos_train_z1=[0.5] n_test=[150]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_02_04" p_pos_train_z0=[0.2] p_pos_train_z1=[0.4] n_test=[150]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_03_06" p_pos_train_z0=[0.3] p_pos_train_z1=[0.6] n_test=[150]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_02_06" p_pos_train_z0=[0.2] p_pos_train_z1=[0.6] n_test=[150]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_05_03" p_pos_train_z0=[0.5] p_pos_train_z1=[0.3] n_test=[150] n_test_error=[5]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_05_01" p_pos_train_z0=[0.5] p_pos_train_z1=[0.1] n_test=[150] n_test_error=[15]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_05_03" p_pos_train_z0=[0.5] p_pos_train_z1=[0.3] n_test=[150] n_test_error=[20] alpha_test=[0,5,0.05]

# python3 runs_xiruo/step103_runExp_SingleLabel.py with proj_name="exp_SingleHead_01_05" p_pos_train_z0=[0.1] p_pos_train_z1=[0.5] n_test=[150]
# python3 runs_xiruo/step103_runExp_SingleLabel.py with proj_name="exp_SingleHead_02_04" p_pos_train_z0=[0.2] p_pos_train_z1=[0.4] n_test=[150]
# python3 runs_xiruo/step103_runExp_SingleLabel.py with proj_name="exp_SingleHead_03_06" p_pos_train_z0=[0.3] p_pos_train_z1=[0.6] n_test=[150]
# python3 runs_xiruo/step103_runExp_SingleLabel.py with proj_name="exp_SingleHead_05_03" p_pos_train_z0=[0.5] p_pos_train_z1=[0.3] n_test=[150] n_test_error=[5]
# python3 runs_xiruo/step103_runExp_SingleLabel.py with proj_name="exp_SingleHead_05_01" p_pos_train_z0=[0.5] p_pos_train_z1=[0.1] n_test=[150] n_test_error=[15]
# python3 runs_xiruo/step103_runExp_SingleLabel.py with proj_name="exp_SingleHead_05_03" p_pos_train_z0=[0.5] p_pos_train_z1=[0.3] n_test=[150] n_test_error=[20] alpha_test=[0,5,0.05]


# Adversarial Model: Gradient Reversal
# python3 runs_xiruo/step102_runExp_TwoHeads.py with \
#     proj_name="exp_GradientReversal_01_05" grad_reverse=True num_epochs=7 lr=5e-5 \
#     pretrained="mental/mental-bert-base-uncased" \
#     p_pos_train_z0=[0.1] p_pos_train_z1=[0.5] n_test=[150]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with \
#     proj_name="exp_GradientReversal_02_04" grad_reverse=True num_epochs=7 lr=5e-5 \
#     pretrained="mental/mental-bert-base-uncased" \
#     p_pos_train_z0=[0.2] p_pos_train_z1=[0.4] n_test=[150]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with \
#     proj_name="exp_GradientReversal_03_06" grad_reverse=True num_epochs=7 lr=5e-5 \
#     pretrained="mental/mental-bert-base-uncased" \
#     p_pos_train_z0=[0.3] p_pos_train_z1=[0.6] n_test=[150]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with \
#     proj_name="exp_GradientReversal_02_06" grad_reverse=True num_epochs=7 lr=5e-5 \
#     pretrained="mental/mental-bert-base-uncased" \
#     p_pos_train_z0=[0.2] p_pos_train_z1=[0.6] n_test=[150]

# python3 runs_xiruo/step102_runExp_TwoHeads.py with \
#     proj_name="exp_GradientReversal_05_01" grad_reverse=True num_epochs=7 lr=5e-5 \
#     pretrained="mental/mental-bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.1] n_test=[150] n_test_error=[15]
# python3 runs_xiruo/step102_runExp_TwoHeads.py with \
#     proj_name="exp_GradientReversal_05_03" grad_reverse=True num_epochs=7 lr=5e-5 \
#     pretrained="mental/mental-bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.3] n_test=[150] n_test_error=[20] alpha_test=[0,5,0.05]


#===== SHAC
# single Head Model
# python3 runs_xiruo/step103_runExp_SingleLabel_SHAC.py with \
#     proj_name="exp_SingleHead_SHAC_05_02" \
#     pretrained="bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[150]

# Two Heads model: Grad Reverse
# python3 runs_xiruo/step102_runExp_TwoHeads_SHAC.py with \
#     proj_name="exp_GradientReversal_SHAC_05_02" grad_reverse=True\
#     pretrained="bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[150]

# Two Heads model, auxilliary task
# python3 runs_xiruo/step102_runExp_TwoHeads_SHAC.py with \
#     proj_name="exp_TwoHeads_SHAC_05_02" \
#     pretrained="bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[150]

# single Head Model
# python3 runs_xiruo/step103_runExp_SingleLabel_SHAC.py with \
#     proj_name="exp_SingleHead_SHAC_n500_05_02" \
#     pretrained="bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[500] \
#     alpha_test=[0,5,0.1]

# single Head Model
# python3 runs_xiruo/step103_runExp_SingleLabel_SHAC.py with \
#     proj_name="exp_SingleHead_SHAC_ClinicalBERT_n500_05_02" \
#     pretrained="medicalai/ClinicalBERT" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[500] \
#     alpha_test=[0,5,0.1]

# python3 runs_xiruo/step105_runExp_SingleLabel_UID_SHAC.py with \
#     proj_name="exp_SingleHead_UID_SHAC_n500_05_02" \
#     pretrained="bert-base-uncased" \
#     p_pos_train_z0=[0.5] p_pos_train_z1=[0.2] n_test=[500] \
#     alpha_test=[0,5,0.1]



    
    
# p_pos_train_z0="[0, 1, 0.1]"\
# p_pos_train_z1="[0, 1, 0.1]" \
# p_mix_z1="[0, 1, 0.05]" \


# p_pos_train_z0='[0.2, 0.6, 0.2]' \
# p_pos_train_z1="[0.2, 0.6, 0.2]" \
# p_mix_z1="[0.2, 0.6, 0.2]" \
    
    
python3 runs_xiruo/step103_runExp_SingleLabel_SHAC_BalanceAlpha.py with \
    proj_name="exp_SingleHead_SHAC_LoRA_BERT_C_ylimited_n200" \
    pretrained="bert-base-uncased" \
    p_pos_train_z0="[0, 1, 0.1]"\
    p_pos_train_z1="[0, 1, 0.1]" \
    p_mix_z1="[0, 1, 0.05]" \
    numvals=129 \
    base="1.01" \
    n_test=[200] \
    num_epochs=3 \
    lr="1e-4"