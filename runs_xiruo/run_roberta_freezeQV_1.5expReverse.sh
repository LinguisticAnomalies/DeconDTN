### 566, 3636, 6621
### 1152, 6114, 11063

dt=SHAC
# dt=HateSpeech

# python BERT_finetuning_freezeQV.py --CombinationIdx=11063 --model_name='roberta-base' --dataset=$dt --toPredict='Target' --gpu='1'
# python BERT_finetuning_freezeQV.py --CombinationIdx=11063 --model_name='roberta-base' --dataset=$dt --toPredict='Source' --gpu='1'
# python BERT_finetuning_freezeQV.py --CombinationIdx=11063 --model_name='roberta-base' --dataset=$dt --toPredict='Source' --reverseLabel --gpu='1'



# # ### 6621
# # isample=566
# # isample=3636

# for isample in 6621 3636 566
# for isample in 3636 566
# # for isample in 1152 6114 11063
# do
# for i in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0
# do 
# echo "================\n\nRunning Exp $i \n\n ~~~~~"
# python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Reverse-Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/" --gpu='1' --lambda1=1.5 --lambda2=$i
# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/set-$isample-epoch3-lambda1_1.5-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/" --gpu="1" --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/inference_set-$isample-epoch3-lambda1_1.5-lambda2_$i-added" --output_dir="../output/roberta/'$dt'_ReverseSource/" --nRuns=5 --dataset=$dt
# done
# done

# for isample in 566 3636 6621
# # for isample in 1152 6114 11063
# do
# for i in 0.0 1.0
# do 
# echo "================\n\nRunning Exp Base 1.0-$i \n\n ~~~~~"
# python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Reverse-Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/" --gpu='1' --lambda1=1.0 --lambda2=$i
# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/set-$isample-epoch3-lambda1_1.0-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/" --gpu="1" --device="cuda:0" --batch_size=8 --dataset=$dt
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/inference_set-$isample-epoch3-lambda1_1.0-lambda2_$i-added" --output_dir="../output/roberta/'$dt'_ReverseSource/" --nRuns=5 --dataset=$dt
# done
# done



#########==============   TEMP!!!


# isample=6621
ilambda1=1.0

for isample in 11063
do
for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do 
echo "================\n\nRunning Exp Base 1.0-$i \n\n ~~~~~"
python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Reverse-Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/" --gpu='2' --lambda1=1.0 --lambda2=$i
python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/set-$isample-epoch3-lambda1_1.0-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/" --gpu="2" --device="cuda:0" --batch_size=8 --dataset=$dt
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/inference_set-$isample-epoch3-lambda1_1.0-lambda2_$i-added" --output_dir="../output/roberta/'$dt'_ReverseSource/" --nRuns=5 --dataset=$dt
done
done

