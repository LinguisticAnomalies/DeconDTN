python run_llama2_general.py --dataset="CD" --model_size=7 --CombinationIdx=6621 --lora_r=8 --quantization --toPredict='Target' --nTest=200 --batchSize=4 --gpu="0" --device="cuda:0"
python run_llama2_general.py --dataset="CD" --model_size=7 --CombinationIdx=6621 --lora_r=8 --quantization --toPredict='Source' --nTest=200 --batchSize=4 --gpu="0" --device="cuda:0"
python run_llama2_general.py --dataset="CD" --model_size=7 --CombinationIdx=6621 --lora_r=8 --quantization --toPredict='Source' --nTest=200 --batchSize=4 --gpu="0" --device="cuda:0" --reverseLabel



python run_llama2_general.py --dataset="CD" --model_size=7 --CombinationIdx=3636 --lora_r=8 --quantization --toPredict='Target' --nTest=200 --batchSize=4 --gpu="0" --device="cuda:0"
python run_llama2_general.py --dataset="CD" --model_size=7 --CombinationIdx=3636 --lora_r=8 --quantization --toPredict='Source' --nTest=200 --batchSize=4 --gpu="0" --device="cuda:0"
