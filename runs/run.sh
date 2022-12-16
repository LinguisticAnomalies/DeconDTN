#!/bin/bash



# python3 runs/step101_runExp.py with proj_name="exp_01_05" p_pos_train_z0=[0.1] p_pos_train_z1=[0.5] n_test=[150]
# python3 runs/step101_runExp.py with proj_name="exp_02_04" p_pos_train_z0=[0.2] p_pos_train_z1=[0.4] n_test=[150]
# python3 runs/step101_runExp.py with proj_name="exp_03_06" p_pos_train_z0=[0.3] p_pos_train_z1=[0.6] n_test=[150]

# python3 runs/step101_runExp.py with proj_name="exp_01_05_N100" p_pos_train_z0=[0.1] p_pos_train_z1=[0.5] p_mix_z1=[0.05,0.99,0.05] n_test=[100] n_valid_high=1
# python3 runs/step101_runExp.py with proj_name="exp_02_04_N100" p_pos_train_z0=[0.2] p_pos_train_z1=[0.4] p_mix_z1=[0.05,0.99,0.05] n_test=[100] n_valid_high=1
# python3 runs/step101_runExp.py with proj_name="exp_03_06_N100" p_pos_train_z0=[0.3] p_pos_train_z1=[0.6] p_mix_z1=[0.05,0.99,0.05] n_test=[100] n_valid_high=1
# python3 runs/step101_runExp.py with proj_name="exp_05_05_N100" p_pos_train_z0=[0.5] p_pos_train_z1=[0.5] p_mix_z1=[0.05,0.99,0.05] n_test=[100] n_valid_high=1

python3 runs/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_01_05" p_pos_train_z0=[0.1] p_pos_train_z1=[0.5] n_test=[150]
python3 runs/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_02_04" p_pos_train_z0=[0.2] p_pos_train_z1=[0.4] n_test=[150]
python3 runs/step102_runExp_TwoHeads.py with proj_name="exp_TwoHeads_03_06" p_pos_train_z0=[0.3] p_pos_train_z1=[0.6] n_test=[150]