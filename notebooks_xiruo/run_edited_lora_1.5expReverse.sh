
# # ### SHAC: 1.5 Exp. One Side (RS) = 11063
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='0.25' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='0.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='0.75' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='1.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='1.25' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='1.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='1.75' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='2.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='2.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-11063-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-11063-quantization-epoch3-llama-2-7B-loraR-8"  --quantization --gpu="1" --lambda1='1.5' --lambda2='3.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-11063-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# #######
# ### SHAC: 1.5 Exp. One Side (RS) = 1152
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='0.25' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='0.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='0.75' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='1.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='1.25' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='1.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='1.75' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='2.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"



# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='2.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-1152-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-1152-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='3.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-1152-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"


# ### SHAC: 1.5 Exp. One Side (RS) = 6114
# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='0.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"


# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='0.25' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.25-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='0.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='0.75' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_0.75-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='1.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='1.25' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.25-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='1.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='1.75' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_1.75-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='2.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='2.5' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_2.5-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-6114-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-6114-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.5' --lambda2='3.0' --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-6114-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.5-lambda2_3.0-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"



# #### 1.0 - 0.0 & 1.0 - 1.0
# for i in 0.0 1.0
# do
# for isample in 1152 6114 11063
# do
# echo "================\n\n(Reverse Source) Running on Sample $isample Exp 1.0-$i \n\n ~~~~~"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_SHAC/n200/set-$isample-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_SHAC/n200/Reverse-Source-set-$isample-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.0' --lambda2=$i --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_SHAC/n200/Weights_ReverseSource/set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="SHAC"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_SHAC/n200/Inferences_ReverseSource/inference_set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added" --output_dir="../output/tmpData/SHAC_ReverseSource/" --nRuns=5 --dataset="SHAC"
# done
# done


# dt=SHAC
# ## 1.0 Exp
# #### 1.0 - 0.0 & 1.0 - 1.0
# for i in 0.0 1.0
# do
# for isample in 1152 6114 11063
# do
# echo "================\n\n(Reverse Source) Running on Sample $isample Exp 1.0-$i \n\n ~~~~~"

# python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_$dt/n200/set-$isample-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="/bime-munin/xiruod/llama2_$dt/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_$dt/n200/Reverse-Source-set-$isample-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.0' --lambda2=$i --cpuOps
# python run_edited_weights_InferenceOnly.py --weightsEdited="/bime-munin/xiruod/llama2_$dt/n200/Weights_ReverseSource/set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/llama2_$dt/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="$dt"
# python run_edited_weights_Eval.py --inferencePathPrefix="/bime-munin/xiruod/llama2_$dt/n200/Inferences_ReverseSource/inference_set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added" --output_dir="../output/tmpData/$dt_ReverseSource/" --nRuns=5 --dataset="$dt"
# done
# done


###============  New Design!!


## Change Weight Dir
# dt=CD
# wtdir=/home/NETID/xiruod/Downloads/TMPllama2/
# wtdir=/bime-munin/xiruod/
## 1.0 Exp
#### 1.0 - 0.0 & 1.0 - 1.0
# dt=SHAC

for isample in 1152 

do
# for i in 0.2 0.4 0.6 0.8 1.2 1.4 1.6 1.8 2.0
for i in 1.6

do
echo "================\n\n(Reverse Source) Running on Sample $isample Exp 1.0-$i \n\n ~~~~~"

python run_editing_weights_loraMerge.py --target_model_id="/bime-munin/xiruod/llama2_$dt/n200/set-$isample-quantization-epoch3-llama-2-7B-loraR-8" --weightsEditedDir="$wtdir/llama2_$dt/n200/Weights_ReverseSource/" --source_model_id="/bime-munin/xiruod/llama2_$dt/n200/Reverse-Source-set-$isample-quantization-epoch3-llama-2-7B-loraR-8" --quantization --gpu="1" --lambda1='1.0' --lambda2=$i --cpuOps
python run_edited_weights_InferenceOnly.py --weightsEdited="$wtdir/llama2_$dt/n200/Weights_ReverseSource/set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added.pth" --output_dir="$wtdir/llama2_$dt/n200/Inferences_ReverseSource/" --quantization --gpu="1" --device="cuda:0" --batch_size=8 --cpuOps --dataset="$dt"
rm $wtdir/llama2_$dt/n200/Weights_ReverseSource/*
python run_edited_weights_Eval.py --inferencePathPrefix="$wtdir/llama2_$dt/n200/Inferences_ReverseSource/inference_set-$isample-quantization-epoch3-llama-2-7B-loraR-8-lambda1_1.0-lambda2_$i-added" --output_dir="../output/tmpData/${dt}_ReverseSource/" --nRuns=5 --dataset="$dt"
done
done