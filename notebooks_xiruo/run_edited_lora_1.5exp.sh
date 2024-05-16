
## SHAC: 1.5 Exp. One Side (S) = 1152
python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.25' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.75' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.25' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.75' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='2.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='2.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='3.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

#######
### SHAC: 1.5 Exp. One Side (S) = 11063

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.25' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.75' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.25' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.75' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='2.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"



python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='2.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='3.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


### SHAC: 1.5 Exp. One Side (S) = 6114

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.25' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='0.75' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.25' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='1.75' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='2.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"


python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='2.5' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights/" --quantization --gpu="0" --lambda1='1.5' --lambda2='3.0' --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/" --quantization --gpu="0" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC/" --nRuns=5 --dataset="SHAC"