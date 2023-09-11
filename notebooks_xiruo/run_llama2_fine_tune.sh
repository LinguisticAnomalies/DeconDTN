# using llama_finetuning.py from llama recipies, on alpaca dataset
python llama_finetuning.py --use_peft --peft_method lora --model_name /bime-munin/llama2_hf/llama-2-7b_hf -
-output_dir /bime-munin/xiruod/testLLaMa --dataset alpaca_dataset --num_epochs 1

