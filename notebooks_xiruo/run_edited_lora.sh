python run_edited_lora_SHAC.py --adapterDir="../output/tmpData/set-1355-quantization-epoch3-llama-2-7B-loraR-8/delta/" --output_dir="../output/tmpData/LoraPredict" --percent=5 --quantization --gpu="0"

# python run_edited_lora_SHAC.py --adapterDir="../output/tmpData/set-1355-quantization-epoch3-llama-2-7B-loraR-8/delta/" --output_dir="../output/tmpData/LoraPredict" --percent=10 --quantization --gpu="2" --batch_size=16

python run_edited_lora_SHAC.py --adapterDir="/bime-munin/xiruod/llama2_SHAC/n500/set-1355-quantization-epoch3-llama-2-7B-loraR-8/" --output_dir="../output/tmpData/LoraPredict_original_Target" --percent=10 --quantization --gpu="2" --batch_size=16