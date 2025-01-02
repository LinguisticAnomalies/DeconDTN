# ### 566, 3636, 6621
# ### 1152, 6114, 11063

# dt=CD
dt=HateSpeech
# dt=SHAC

python BERT_finetuning_freezeQV.py --CombinationIdx=3636 --model_name='roberta-base' --dataset=$dt --toPredict='Target'  --gpu='3'
python BERT_finetuning_freezeQV.py --CombinationIdx=3636 --model_name='roberta-base' --dataset=$dt --toPredict='Source'  --gpu='3'
python BERT_finetuning_freezeQV.py --CombinationIdx=3636 --model_name='roberta-base' --dataset=$dt --toPredict='Source' --reverseLabel  --gpu='3'

dt=SHAC
python BERT_finetuning_freezeQV.py --CombinationIdx=11063 --model_name='roberta-base' --dataset=$dt --toPredict='Target'  --gpu='2'
python BERT_finetuning_freezeQV.py --CombinationIdx=11063 --model_name='roberta-base' --dataset=$dt --toPredict='Source'  --gpu='2'
python BERT_finetuning_freezeQV.py --CombinationIdx=11063 --model_name='roberta-base' --dataset=$dt --toPredict='Source' --reverseLabel  --gpu='2'







#####-------- 1.0 Exp
gpu='1'

# dt=CD
# dt=HateSpeech
dt=SHAC



ilambda1=1.0

# for isample in 566 6621
for isample in 1152 11063
do
for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
echo "================\n\nRunning DT $dt Sample $isample Exp $i \n\n ~~~~~"

python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/" --gpu=$gpu --lambda1=$ilambda1 --lambda2=$i
python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/set-$isample-epoch3-lambda1_$ilambda1-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
rm /bime-munin/xiruod/roberta-base_$dt/n200/Weights/*
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/inference_set-$isample-epoch3-lambda1_$ilambda1-lambda2_$i-added" --output_dir="../output/roberta/$dt/" --nRuns=5 --dataset=$dt
done
done




#####-------- 1.0 Exp -- Reverse

gpu='3'

dt=CD


ilambda1=1.0

for isample in 566 6621
do
for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do 
echo "================\n\nRunning Exp Base 1.0-$i \n\n ~~~~~"
python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Reverse-Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/" --gpu=$gpu --lambda1=1.0 --lambda2=$i
python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/set-$isample-epoch3-lambda1_1.0-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
rm /bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/*
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/inference_set-$isample-epoch3-lambda1_1.0-lambda2_$i-added" --output_dir="../output/roberta/'$dt'_ReverseSource/" --nRuns=5 --dataset=$dt
done
done



################ Run Separately
##-------------------------------


################ Do Inference

#####-------- 1.0 Exp
gpu='1'

# dt=CD
# dt=HateSpeech
# dt=SHAC



ilambda1=1.0

for dt in CD HateSpeech
do
for isample in 566 6621
# for dt in SHAC
# do
# for isample in 1152 11063
do
for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
echo "================\n\nRunning DT $dt Sample $isample Exp $i \n\n ~~~~~"

python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/" --gpu=$gpu --lambda1=$ilambda1 --lambda2=$i
python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/set-$isample-epoch3-lambda1_$ilambda1-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
rm /bime-munin/xiruod/roberta-base_$dt/n200/Weights/*
done
done
done




#####-------- 1.0 Exp -- Reverse

gpu='3'

dt=CD


ilambda1=1.0

for dt in CD HateSpeech
do
for isample in 566 6621
do
for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do 
echo "================\n\nRunning DT $dt Sample $isample Exp $i \n\n ~~~~~"
python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Reverse-Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/" --gpu=$gpu --lambda1=1.0 --lambda2=$i
python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/set-$isample-epoch3-lambda1_1.0-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
rm /bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/*
done
done
done





## Inference for alpha=1
gpu='0'

dt=CD


ilambda1=1.0


for isample in 3636
do
for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do 

# python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/" --gpu=$gpu --lambda1=$ilambda1 --lambda2=$i
# python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/set-$isample-epoch3-lambda1_$ilambda1-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
# rm /bime-munin/xiruod/roberta-base_$dt/n200/Weights/*

python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Reverse-Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/" --gpu=$gpu --lambda1=1.0 --lambda2=$i
python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/set-$isample-epoch3-lambda1_1.0-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt
rm /bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/*

done
done








##############  Do Eval


#####-------- 1.0 Exp
# dt=CD
dt=HateSpeech
# dt=SHAC



ilambda1=1.0

# for isample in 566 6621
# for isample in 1152 11063

for isample in 3636
# for isample in 6114
do
for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
echo "================\n\nRunning DT $dt Sample $isample Exp $i \n\n ~~~~~"
# python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/inference_set-$isample-epoch3-lambda1_$ilambda1-lambda2_$i-added" --output_dir="../output/roberta/$dt/" --nRuns=5 --dataset=$dt &

python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/inference_set-$isample-epoch3-lambda1_1.0-lambda2_$i-added" --output_dir="../output/roberta/'$dt'_ReverseSource/" --nRuns=5 --dataset=$dt &

done
wait
done





### run 566 & 6621 together
#####-------- 1.0 Exp
dt=CD
# dt=HateSpeech
# dt=SHAC



ilambda1=1.0


for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
echo "================\n\nRunning DT $dt Sample Exp $i \n\n ~~~~~"
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/inference_set-1152-epoch3-lambda1_$ilambda1-lambda2_$i-added" --output_dir="../output/roberta/$dt/" --nRuns=5 --dataset=$dt &
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/inference_set-11063-epoch3-lambda1_$ilambda1-lambda2_$i-added" --output_dir="../output/roberta/$dt/" --nRuns=5 --dataset=$dt &

python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/inference_set-1152-epoch3-lambda1_1.0-lambda2_$i-added" --output_dir="../output/roberta/'$dt'_ReverseSource/" --nRuns=5 --dataset=$dt &
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences_ReverseSource/inference_set-11063-epoch3-lambda1_1.0-lambda2_$i-added" --output_dir="../output/roberta/'$dt'_ReverseSource/" --nRuns=5 --dataset=$dt

done




#######----- TEMP


#####-------- 1.0 Exp
gpu='3'

dt=CD




ilambda1=1.0

for isample in 6621
do
for i in 0.0 1.0
do
echo "================\n\nRunning DT $dt Sample $isample Exp $i \n\n ~~~~~"

# python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/" --gpu=$gpu --lambda1=$ilambda1 --lambda2=$i

python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Reverse-Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights_ReverseSource/" --gpu=$gpu --lambda1=1.0 --lambda2=$i

done
done
