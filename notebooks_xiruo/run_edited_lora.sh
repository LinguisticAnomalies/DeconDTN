

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




############### Just Finished
# python run_editing_lora.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_FroNorm/set-1355-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="0,2" --targetFroNorm

# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_FroNorm/set-1355-quantization-epoch3-llama-2-13B-loraR-8/delta" --output_dir="../output/tmpData/LoraAdapters_TargetFroNorm/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=8


# python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/LoraAdapters_FroNorm/set-1355-quantization-epoch3-llama-2-7B-loraR-8/delta" --output_dir="../output/tmpData/LoraAdapters_TargetFroNorm/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=32

# python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-7B-loraR-8-gamma_1-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_1_Added/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=32


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-13B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n500/Weights/" --quantization --gpu="0,1,2" --gamma=1

python run_edited_weights_SHAC.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n500/Weights/set-1355-quantization-epoch3-llama-2-13B-loraR-8-gamma_1-added.pth" --output_dir="../output/tmpData/WeightsEdited_Gamma_1_Added/" --percent=15 --quantization --gpu="0,2" --device="cuda:0" --batch_size=8

#~~~ Cognitive Distortion
# python avh_Mistral.py --job='avh_noDOT_All_NoExample_Mistral'

############### Ongoing

python run_noDoT_benchmark_Mistral.py


############### To Be Scheduled






# TODO: Weight Normalization: vector?? --> Frob Matrix Norm? or LayerNorm, BatchNorm 
