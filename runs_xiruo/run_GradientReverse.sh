

## 566, 3636, 6621
## 1152, 6114, 11063

# dt=CD
# dt=HateSpeech
# dt=SHAC




ep=100
gpu='2'
for dt in 'CD'
do
for isample in 6621
# for dt in 'SHAC'
# do
# for isample in 1152 6114 11063

do
echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"
python RoBERTa_GradientReverse.py --dataset=$dt --CombinationIdx=$isample --num_train_epochs=$ep --batchSize=16 --gpu=$gpu --device="cuda"

python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --GradientReverse=True --weightsEdited="/bime-munin/xiruod/GradientReverse/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/GradientReverse/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/GradientReverse/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-GradientReverse-epoch$ep/" --nRuns=5 --dataset=$dt
done
done



