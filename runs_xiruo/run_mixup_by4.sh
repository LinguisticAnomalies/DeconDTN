###### mixup by 4
# for dt in 'CD' 'HateSpeech'
# do
# for isample in 566 6621
# do

for dt in 'HateSpeech'
do
for isample in 6621
do

echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

python mixup_by4.py --dataset=$dt --CombinationIdx=$isample --num_train_epochs=20

done
done


for dt in 'SHAC'
do
for isample in 1152 11063
do
echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

python mixup_by4.py --dataset=$dt --CombinationIdx=$isample --num_train_epochs=6

done
done



augm='mixup-by4'

for dt in 'CD' 'HateSpeech'
do
for isample in 566 3636 6621

# for dt in 'SHAC'
# do
# for isample in 1152 6114 11063
# ep=6
do
ep=20

echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/set-$isample-epoch$ep-mixupAlpha_4/pytorch_model.bin" --output_dir="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep-mixupAlpha_4/" --gpu="3" --device="cuda:0" --batch_size=8 --dataset=$dt
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep-mixupAlpha_4/inference_set-$isample-epoch$ep-mixupAlpha_4" --output_dir="../output/roberta/$dt-FullFT-$augm-epoch$ep-mixupAlpha_4/" --nRuns=5 --dataset=$dt
done
done




# ep=60

# # # for dt in 'CD' 'HateSpeech'
# # # do
# # # for isample in 566 3636 6621

# for dt in 'SHAC'
# do
# for isample in 1152 6114 11063
# do


# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# python coral_mmd_gdro.py --dataset=$dt --CombinationIdx=$isample --nRuns=$ep --batchSize=32 --device="cuda"


# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/GDRO/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/GDRO/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu="3" --device="cuda:0" --batch_size=8 --dataset=$dt

# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/GDRO/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-GDRO-epoch$ep/" --nRuns=5 --dataset=$dt


# done
# done





# ######## MMD

# ## 566, 3636, 6621
# ## 1152, 6114, 11063

# dt=CD
# dt=HateSpeech
# dt=SHAC




# ep=100

# # for dt in 'CD' 'HateSpeech'
# # do
# # for isample in 566 3636 6621
# # for dt in 'SHAC'
# # do
# # for isample in 1152 6114 11063
# # do

# for dt in 'HateSpeech' 
# do
# for isample in 566 6621
# do
# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/MMD/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/MMD/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu="3" --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/MMD/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-MMD-epoch$ep/" --nRuns=5 --dataset=$dt
# done
# done


# for dt in 'SHAC' 
# do
# for isample in 1152 11063
# do
# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/MMD/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/MMD/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu="3" --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/MMD/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-MMD-epoch$ep/" --nRuns=5 --dataset=$dt
# done
# done



# # augm='mixup'

# # for dt in 'CD' 'HateSpeech'
# # do
# # for isample in 566 3636 6621
# # # for dt in 'SHAC'
# # # do
# # # for isample in 1152 6114 11063
# # do
# # echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

# # python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/set-$isample-epoch20-mixupAlpha_4/pytorch_model.bin" --output_dir="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch20-mixupAlpha_4/" --gpu="3" --device="cuda:0" --batch_size=8 --dataset=$dt
# # python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch20-mixupAlpha_4/inference_set-$isample-epoch20-mixupAlpha_4" --output_dir="../output/roberta/$dt-FullFT-$augm-epoch20-mixupAlpha_4/" --nRuns=5 --dataset=$dt
# # done
# # done



