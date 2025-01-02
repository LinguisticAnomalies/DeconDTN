# dt=HateSpeech
# dt=CD
# isample=566


for dt in CD HateSpeech
do
for isample in 3636 #566 6621
do

python Backdoor_TrainOnFix_and_Inference.py  --CombinationIdx=$isample --model_name='roberta-base' --dataset=$dt  --nTest=200 --transform='binaryUnigram' --runs=1 --output_dir="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_Backdoor/" --output_dir_noBackdoor="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_noBackdoor/"
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_Backdoor/inference_set-$isample-irun-0" --output_dir="../output/backdoor/$dt/backdoor/" --nRuns=5 --dataset=$dt &
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_noBackdoor/inference_set-$isample-irun-0" --output_dir="../output/backdoor/$dt/noBackdoor/" --nRuns=5 --dataset=$dt
done
done



for dt in SHAC
do
for isample in 6114 # 1152 11063
do

python Backdoor_TrainOnFix_and_Inference.py  --CombinationIdx=$isample --model_name='roberta-base' --dataset=$dt  --nTest=200 --transform='binaryUnigram' --runs=1 --output_dir="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_Backdoor/" --output_dir_noBackdoor="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_noBackdoor/"
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_Backdoor/inference_set-$isample-irun-0" --output_dir="../output/backdoor/$dt/backdoor/" --nRuns=5 --dataset=$dt &
python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_noBackdoor/inference_set-$isample-irun-0" --output_dir="../output/backdoor/$dt/noBackdoor/" --nRuns=5 --dataset=$dt

done
done