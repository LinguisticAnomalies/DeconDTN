

## 566, 3636, 6621
## 1152, 6114, 11063

# dt=CD
# dt=HateSpeech
# dt=SHAC




# ep=3
ep=200
gpu='3'
for dt in 'HateSpeech'
# for dt in 'CD' 'HateSpeech'
do
# for isample in 6621 566 3636
for isample in 6621 566
# for dt in 'SHAC'
# do
# for isample in 1152 6114 11063

# do
# echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"
# python RoBERTa_GradientReverse.py --dataset=$dt --CombinationIdx=$isample --num_train_epochs=$ep --batchSize=16 --gpu=$gpu --device="cuda"

# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --GradientReverse=True --weightsEdited="/bime-munin/xiruod/GradientReverse/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/GradientReverse/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/GradientReverse/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-GradientReverse-epoch$ep/" --nRuns=5 --dataset=$dt
# done
# done



do
echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"
python RoBERTa_GradientReverse_ForDomainBed.py --dataset=$dt --CombinationIdx=$isample --num_train_epochs=$ep --batchSize=16 --gpu=$gpu --device="cuda"

echo "\n\nDT $dt Sample $isample Exp: Finished Training \n\n"

python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --GradientReverse=True --weightsEdited="/bime-munin/xiruod/GradientReverse_ForDomainBed/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/GradientReverse_ForDomainBed/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt

echo "\n\nDT $dt Sample $isample Exp: Finished Inference \n\n"

python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/GradientReverse_ForDomainBed/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-GradientReverse_ForDomainBed-epoch$ep/" --nRuns=5 --dataset=$dt

echo "\n\nDT $dt Sample $isample Exp: Finished Eval \n\n"

done
done




### TO DELETE!

ep=100
gpu='3'

for dt in 'SHAC'
do
for isample in 1152 6114 11063


do
echo "================\n\nRunning DT $dt Sample $isample Exp \n\n ~~~~~"
python RoBERTa_GradientReverse_ForDomainBed.py --dataset=$dt --CombinationIdx=$isample --num_train_epochs=$ep --batchSize=16 --gpu=$gpu --device="cuda"

python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --GradientReverse=True --weightsEdited="/bime-munin/xiruod/GradientReverse_ForDomainBed/roberta-base_$dt/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/GradientReverse_ForDomainBed/roberta-base_$dt/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/GradientReverse_ForDomainBed/roberta-base_$dt/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-GradientReverse_ForDomainBed-epoch$ep/" --nRuns=5 --dataset=$dt
done
done
