### 566, 3636, 6621
### 1152, 6114, 11063

# dt=CD
# dt=HateSpeech
# dt=SHAC

# # augm='ReSample'
# # augm='EDA'
# augm='noaug'

# # for dt in 'CD' 'HateSpeech'
# # do
# # for isample in 566 3636 6621
# for dt in 'SHAC'
# do
# for isample in 1152 6114 11063
# do
# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# python BERT_finetuning_AugUp.py --CombinationIdx=$isample --model_name='roberta-base' --dataset=$dt --augMethod=$augm  --gpu=$gpu
# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/set-$isample-epoch3/pytorch_model.bin" --output_dir="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences/inference_set-$isample-epoch3" --output_dir="../output/roberta/$dt-FullFT-$augm/" --nRuns=5 --dataset=$dt
# done
# done

### Main
# gpu=2
# ###### epoch 20
# # dt=CD
# # dt=HateSpeech
# # dt=SHAC

# # augm='AllEqual-ReSample'
# # augm='EDA_ReturnAll'
# # augm='EDA'
# # augm='noaug'
# augm='ReSample_by4'

# ep=20

# for dt in 'CD'
# do
# for isample in 3636 6621
# # for dt in 'CD' 'HateSpeech'
# # do
# # for isample in 566 3636 6621
# # for dt in 'SHAC'
# # do
# # for isample in 1152 6114 11063
# do
# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# python BERT_finetuning_AugUp.py --CombinationIdx=$isample --model_name='roberta-base' --dataset=$dt --augMethod=$augm  --gpu=$gpu --num_train_epochs=$ep
# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-FullFT-$augm-epoch$ep/" --nRuns=5 --dataset=$dt


# done
# done





# ## dups from above.. for running SHAC  ## epoch=6 for SHAC
# ep=6
# for dt in 'SHAC'
# do
# for isample in 1152 11063
# do
# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# python BERT_finetuning_AugUp.py --CombinationIdx=$isample --model_name='roberta-base' --dataset=$dt --augMethod=$augm  --gpu=$gpu --num_train_epochs=$ep
# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-FullFT-$augm-epoch$ep/" --nRuns=5 --dataset=$dt
# done
# done




################# mixup

# augm='mixup'

# for dt in 'CD' 'HateSpeech'
# do
# for isample in 566 3636 6621
# # for dt in 'SHAC'
# # do
# # for isample in 1152 6114 11063
# do
# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# ep=6

# python mixup_RoBERTa.py --dataset=$dt --CombinationIdx=$isample --num_train_epochs=$ep --gpu=$gpu

# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/set-$isample-epoch$ep-mixupAlpha_4/pytorch_model.bin" --output_dir="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep-mixupAlpha_4/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep-mixupAlpha_4/inference_set-$isample-epoch$ep-mixupAlpha_4" --output_dir="../output/roberta/$dt-FullFT-$augm-epoch$ep-mixupAlpha_4/" --nRuns=5 --dataset=$dt
# done
# done






# #### Test for Yongsen's Settings
# augm='noaug'

# gpu=2
# ep=40

# for dt in 'HateSpeech'
# do
# for isample in 6621

# do
# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# python BERT_finetuning_AugUp.py --CombinationIdx=$isample --model_name='roberta-base' --dataset=$dt --augMethod=$augm  --gpu=$gpu --num_train_epochs=$ep
# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-FullFT-$augm-epoch$ep/" --nRuns=5 --dataset=$dt


# done
# done
