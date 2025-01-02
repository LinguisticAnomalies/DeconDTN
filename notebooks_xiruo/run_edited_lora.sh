

############### Unfinished
# python run_edited_lora_SHAC.py --adapterDir="../output/tmpData/set-1355-quantization-epoch3-llama-2-7B-loraR-8/delta/" --output_dir="../output/tmpData/LoraPredict" --percent=5 --quantization --gpu="0"


############### Finished
# python run_edited_lora_SHAC.py --adapterDir="../output/tmpData/set-1355-quantization-epoch3-llama-2-7B-loraR-8/delta/" --output_dir="../output/tmpData/LoraPredict" --percent=15 --quantization --gpu="2" --batch_size=16

# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8/" --output_dir="../output/tmpData/LoraPredict_Original_Target" --percent=15 --quantization --gpu="0" --batch_size=32

# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_TargetNorm/set-1355-quantization-epoch3-llama-2-7B-loraR-8/delta" --output_dir="../output/tmpData/LoraAdapters_TargetNorm/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=32





# python run_editing_lora.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --quantization --gpu="2,1"
# python run_editing_lora.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_TargetNorm/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --quantization --gpu="0,2" --targetNorm
# python run_editing_lora.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_FroNorm/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --quantization --gpu="0,2" --targetFroNorm


# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters/set-1355-quantization-epoch3-llama-2-13B-loraR-8/delta" --output_dir="../output/tmpData/LoraPredict" --percent=15 --quantization --gpu="2,1" --device="cuda:1" --batch_size=8

# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --output_dir="../output/tmpData/LoraPredict_Original_Target" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=8

# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_TargetNorm/set-1355-quantization-epoch3-llama-2-13B-loraR-8/delta" --output_dir="../output/tmpData/LoraAdapters_TargetNorm/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=8

# python run_editing_lora.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_FroNorm/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="0,2" --targetFroNorm

# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_FroNorm/set-1355-quantization-epoch3-llama-2-13B-loraR-8/delta" --output_dir="../output/tmpData/LoraAdapters_TargetFroNorm/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=8


# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_FroNorm/set-1355-quantization-epoch3-llama-2-7B-loraR-8/delta" --output_dir="../output/tmpData/LoraAdapters_TargetFroNorm/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=32

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_1_Added/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=32


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma=1

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-13B-loraR-8-gamma_1-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_1_Added/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=8


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='0.1' --DeltaFinished

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='0.2' --DeltaFinished

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='0.5'

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='0.8' --DeltaFinished



# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_0.1-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_0.1_Added/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_0.2-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_0.2_Added/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_0.5-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_0.5_Added/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=32

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_0.8-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_0.8_Added/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='1.5' --DeltaFinished

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='2.0' --DeltaFinished

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='3.0' --DeltaFinished

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1.5-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_1.5_Added/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_2.0-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_2.0_Added/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_3.0-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_3.0_Added/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16



############### Just Finished

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='1' --lambda1='1' --lambda2='0.5'

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1.0-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_1.0_Lamda1_1.0_Lambda2_0.5_Added/" --percent=15 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1.0-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_1.0_Lamda1_2.0_Lambda2_1.0_Added/" --percent=20 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='1' --lambda1='2' --lambda2='1'

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma='1' --lambda1='1' --lambda2='0'

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1.0-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_1.0_Lamda1_1.0_Lambda2_0.0_Added/" --percent=20 --quantization --gpu="0,1" --device="cuda:0" --batch_size=16






# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="2" --lambda1='2' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0" --lambda1='3' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0" --lambda1='4' --lambda2='2' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0" --lambda1='2' --lambda2='0.5' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0.5' --cpuOps



# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="../output/tmpData/SHAC/" --percent=20 --quantization --gpu="0" --device="cuda:0" --batch_size=16

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="../output/tmpData/SHAC/" --percent=20 --quantization --gpu="0" --device="cuda:0" --batch_size=16

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added.pth" --output_dir="../output/tmpData/SHAC/" --percent=15 --quantization --gpu="0" --device="cuda:0" --batch_size=16

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps
# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --percent=5 --sampleValidSettings --cpuOps --nRuns=5



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='2' --lambda2='1' --cpuOps
# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="1" --device="cuda:0" --batch_size=16 --percent=5 --sampleValidSettings --cpuOps --nRuns=5

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0.5' --cpuOps
# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="2" --device="cuda:0" --batch_size=8  --percent=5 --sampleValidSettings --cpuOps --nRuns=5





# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-2800-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps
# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --percent=5 --sampleValidSettings --cpuOps --nRuns=5

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-2800-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='2' --lambda2='1' --cpuOps
# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --percent=5 --sampleValidSettings --cpuOps --nRuns=5



# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --percent=5 --sampleValidSettings --cpuOps --nRuns=5


# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --percent=5 --sampleValidSettings --cpuOps --nRuns=5



# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --percent=5 --sampleValidSettings --cpuOps --nRuns=5


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-13755-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps
# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --percent=5 --sampleValidSettings --cpuOps --nRuns=5

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-13755-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='2' --lambda2='1' --cpuOps


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='3' --lambda2='1' --cpuOps

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added.pth" --output_dir="../output/tmpData/SHAC/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --percent=5 --sampleValidSettings --cpuOps --nRuns=5








## SHAC

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-2800-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0.5' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0.5' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-13755-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0.5' --cpuOps





# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-2800-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='3' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-13755-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='3' --lambda2='1' --cpuOps


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-2800-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='4' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-13755-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='4' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='4' --lambda2='1' --cpuOps




# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"



# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"




# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"




# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"




# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=16 --cpuOps --dataset="SHAC"




# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"




# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"




# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1355-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-2800-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-13755-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"




# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0.5' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0.5' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0.5' --cpuOps



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='2' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='2' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='2' --lambda2='1' --cpuOps



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='3' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='3' --lambda2='1' --cpuOps

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='3' --lambda2='1' --cpuOps



## HateSpeech
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"




# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"


# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"




# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-6126-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-9870-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n1000/Inferences/inference_set-1874-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"



# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"



# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"



# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"



# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"








# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"





############### Ongoing



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='0' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"





# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='1' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='1' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1' --lambda2='1' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"




# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.5' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.5' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.5' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"







# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='2' --lambda2='1' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='2' --lambda2='1' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='2' --lambda2='1' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.7' --lambda2='0.7' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.7' --lambda2='0.7' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.7' --lambda2='0.7' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.7-lambda2_0.7-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='3' --lambda2='2' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='3' --lambda2='2' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='3' --lambda2='2' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_3.0-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.3' --lambda2='0.3' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.3' --lambda2='0.3' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.3' --lambda2='0.3' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"





# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='4' --lambda2='3' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='4' --lambda2='3' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='4' --lambda2='3' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_4.0-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.1' --lambda2='0.1' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.1' --lambda2='0.1' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.1' --lambda2='0.1' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.1-lambda2_0.1-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.05' --lambda2='0.05' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.05' --lambda2='0.05' --cpuOps
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="2" --lambda1='1.05' --lambda2='0.05' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.05-lambda2_0.05-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



############### To Be Scheduled


# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='1' --lambda2='0' --lambda3='0' --cpuOps
# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='1' --lambda2='0' --lambda3='0' --cpuOps
# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='1' --lambda2='0' --lambda3='0' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-lambda3_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='2' --lambda2='1' --lambda3='0' --cpuOps
# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='2' --lambda2='1' --lambda3='0' --cpuOps
# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='2' --lambda2='1' --lambda3='0' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_1.0-lambda3_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"




# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='2' --lambda2='0' --lambda3='1' --cpuOps
# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='2' --lambda2='0' --lambda3='1' --cpuOps
# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="1" --lambda1='2' --lambda2='0' --lambda3='1' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_2.0-lambda2_0.0-lambda3_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.25' --lambda3='0.25' --cpuOps
# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.25' --lambda3='0.25' --cpuOps
# python run_editing_weights_loraMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.25' --lambda3='0.25' --cpuOps

# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-lambda3_0.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"






##### Two Sources, NormProj
# python run_editing_weights_loraNormProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNormProj/" --quantization --gpu="2" --lambda1='1.5' --cpuOps --projVecName='S'
# python run_editing_weights_loraNormProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNormProj/" --quantization --gpu="2" --lambda1='1.5' --cpuOps --projVecName='S'
# python run_editing_weights_loraNormProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNormProj/" --quantization --gpu="2" --lambda1='1.5' --cpuOps --projVecName='RS'
# python run_editing_weights_loraNormProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNormProj/" --quantization --gpu="2" --lambda1='1.5' --cpuOps --projVecName='RS'




# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNormProj/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_S.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNormProj/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNormProj/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_S.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNormProj/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNormProj/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_RS.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNormProj/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNormProj/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_RS.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNormProj/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNormProj/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_S" --output_dir="../output/tmpData/SHAC_norm_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNormProj/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_S" --output_dir="../output/tmpData/SHAC_norm_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNormProj/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_RS" --output_dir="../output/tmpData/SHAC_norm_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNormProj/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm-Proj_RS" --output_dir="../output/tmpData/SHAC_norm_proj/" --nRuns=5 --dataset="SHAC"





#### Two sources, Projection


# python run_editing_weights_loraProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1' --lambda3='0.5' --cpuOps --projVecName='S'
# python run_editing_weights_loraProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1' --lambda3='0.5' --cpuOps --projVecName='S'
# python run_editing_weights_loraProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.5' --lambda3='1' --cpuOps --projVecName='RS'
# python run_editing_weights_loraProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.5' --lambda3='1' --cpuOps --projVecName='RS'




# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-lambda3_0.5-added-Proj_S.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-lambda3_0.5-added-Proj_S.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_RS.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_RS.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-lambda3_0.5-added-Proj_S" --output_dir="../output/tmpData/SHAC_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-lambda3_0.5-added-Proj_S" --output_dir="../output/tmpData/SHAC_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_RS" --output_dir="../output/tmpData/SHAC_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-lambda3_1.0-added-Proj_RS" --output_dir="../output/tmpData/SHAC_proj/" --nRuns=5 --dataset="SHAC"




### Proj Only

# python run_editing_weights_loraProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/" --quantization --gpu="0" --lambda1='1.3' --lambda2='1' --lambda3='0.3' --cpuOps --projVecName='S'
# python run_editing_weights_loraProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/" --quantization --gpu="0" --lambda1='1.3' --lambda2='1' --lambda3='0.3' --cpuOps --projVecName='S'
# python run_editing_weights_loraProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/" --quantization --gpu="0" --lambda1='1.3' --lambda2='0.3' --lambda3='1' --cpuOps --projVecName='RS'
# python run_editing_weights_loraProjectionMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/" --quantization --gpu="0" --lambda1='1.3' --lambda2='0.3' --lambda3='1' --cpuOps --projVecName='RS'




# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_1.0-lambda3_0.3-added-Proj_S.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_1.0-lambda3_0.3-added-Proj_S.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_RS.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsProj/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_RS.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"



# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_1.0-lambda3_0.3-added-Proj_S" --output_dir="../output/tmpData/SHAC_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_1.0-lambda3_0.3-added-Proj_S" --output_dir="../output/tmpData/SHAC_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_RS" --output_dir="../output/tmpData/SHAC_proj/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesProj/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-lambda2_0.3-lambda3_1.0-added-Proj_RS" --output_dir="../output/tmpData/SHAC_proj/" --nRuns=5 --dataset="SHAC"



#### Norm Only
# python run_editing_weights_loraNormMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/" --quantization --gpu="2" --lambda1='1.5' --cpuOps
# python run_editing_weights_loraNormMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/" --quantization --gpu="2" --lambda1='1.5' --cpuOps
# python run_editing_weights_loraNormMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/" --quantization --gpu="2" --lambda1='1.5' --cpuOps


# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm" --output_dir="../output/tmpData/SHAC_norm/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm" --output_dir="../output/tmpData/SHAC_norm/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-added-Norm" --output_dir="../output/tmpData/SHAC_norm/" --nRuns=5 --dataset="SHAC"




# python run_editing_weights_loraNormMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/" --quantization --gpu="2" --lambda1='1.3' --cpuOps
# python run_editing_weights_loraNormMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/" --quantization --gpu="2" --lambda1='1.3' --cpuOps
# python run_editing_weights_loraNormMerge_TwoSources.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/" --quantization --gpu="2" --lambda1='1.3' --cpuOps


# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/WeightsNorm/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/" --quantization --gpu="2" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"

# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm" --output_dir="../output/tmpData/SHAC_norm/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm" --output_dir="../output/tmpData/SHAC_norm/" --nRuns=5 --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/InferencesNorm/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.3-added-Norm" --output_dir="../output/tmpData/SHAC_norm/" --nRuns=5 --dataset="SHAC"



### SHAC: 1.5 Exp. One Side (S) = 1152
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"







#### Hate Speech Results Collect
### Hate Speech Single Source, One side (S)
#-- 566
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &


#-- 3636
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &



#-- 6621
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"


### Hate Speech Single Source, One side (RS)
#-- 6621
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-6621-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &


#-- 566
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-566-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &

#-- 3636
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_0.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-3636-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_1.0-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"



# for isample in 566
# do
# # for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
# for i in 1.8

# do
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech"
# done
# done




# for isample in 3636
# do
# for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
# do
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech"
# done
# done



##======= TEMP!!!

# for isample in 3636
# do
# for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
# do
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences_ReverseSource/inference_set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added" --output_dir="../output/tmpData/HateSpeech_ReverseSource/" --nRuns=5 --dataset="HateSpeech" &
# done
# done



for isample in 3636
do
for i in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_HateSpeech/n200/Inferences/inference_set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added" --output_dir="../output/tmpData/HateSpeech/" --nRuns=5 --dataset="HateSpeech" &
done
done
