python BERT_finetuning_freezeQV --CombinationIdx=566 --model_name='roberta-base' --dataset='CD' --toPredict='Target'
python BERT_finetuning_freezeQV --CombinationIdx=566 --model_name='roberta-base' --dataset='CD' --toPredict='Source'
python BERT_finetuning_freezeQV --CombinationIdx=566 --model_name='roberta-base' --dataset='CD' --toPredict='Source' --reverseSource


python BERT_editingWeights.py --model_name='roberta-base' --target_model_id='/bime-munin/xiruod/roberta-base_CD/n200/set-566-epoch3' --source_model_id='/bime-munin/xiruod/roberta-base_CD/n200/Source-set-566-epoch3' --weightsEditedDir='/bime-munin/xiruod/roberta-base_CD/n200/Weights/' --lambda1=1.5 --lambda2=1.75

### To Be Added Evaluation!