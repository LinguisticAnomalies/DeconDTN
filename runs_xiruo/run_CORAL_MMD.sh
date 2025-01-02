# ###### GDRO
### GDRO
### GDRO

ep=3
gpu='3'

method='GDRO'

for dt in 'CD' 'HateSpeech'
do
for isample in 566 6621
do



echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

python coral_mmd_gdro.py --dataset=$dt --CombinationIdx=$isample --nRuns=$ep --batchSize=32 --device="cuda" --method=$method


python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/$method/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/$method/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt

python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/$method/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-$method-epoch$ep/" --nRuns=5 --dataset=$dt


done
done




for dt in 'SHAC'
do
for isample in 1152 11063
do


echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

python coral_mmd_gdro.py --dataset=$dt --CombinationIdx=$isample --nRuns=$ep --batchSize=32 --device="cpu" --method=$method


python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/$method/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/$method/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt

python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/$method/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-$method-epoch$ep/" --nRuns=5 --dataset=$dt


done
done




######## MMD

## 566, 3636, 6621
## 1152, 6114, 11063

dt=CD
dt=HateSpeech
dt=SHAC


gpu=3

ep=100



ep=3
gpu='3'

method='MMD'
for dt in 'CD'
do
for isample in 566 6621
do

# for dt in 'SHAC'
# do
# for isample in 1152 6114 11063
# do


echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"

python coral_mmd_gdro.py --dataset=$dt --CombinationIdx=$isample --nRuns=$ep --batchSize=200 --device="cpu" --method="MMD"


python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/$method/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/$method/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt

python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/$method/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-$method-epoch$ep/" --nRuns=5 --dataset=$dt


done
done



