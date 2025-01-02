# ## Old
# python run_Aug_Logistic_and_Backdoor.py --dataset="CD" --transform="Sentence-BERT" --alphaAug=4 --gpu=0 &
# python run_Logistic_and_Backdoor.py --dataset="CD" --transform="Sentence-BERT" --gpu=0 &

# python run_Aug_Logistic_and_Backdoor.py --dataset="HateSpeech" --transform="Sentence-BERT" --alphaAug=4 --gpu=1 &
# python run_Logistic_and_Backdoor.py --dataset="HateSpeech" --transform="Sentence-BERT" --gpu=1 &

# python run_Aug_Logistic_and_Backdoor.py --dataset="SHAC" --transform="Sentence-BERT" --alphaAug=4 --gpu=2 &
# python run_Logistic_and_Backdoor.py --dataset="SHAC" --transform="Sentence-BERT" --gpu=2 &

# python run_AugUp_Resample_Logistic_and_Backdoor.py --dataset="CD" --transform="Sentence-BERT" --gpu=0 &
# python run_AugUp_Resample_Logistic_and_Backdoor.py --dataset="HateSpeech" --transform="Sentence-BERT" --gpu=1 &
# python run_AugUp_Resample_Logistic_and_Backdoor.py --dataset="SHAC" --transform="Sentence-BERT" --gpu=2





# python run_AugUp_mixup_Logistic_and_Backdoor.py --dataset="CD" --transform="Sentence-BERT" --sample=$sample --alphaAug=4 --gpu=0 &
# python run_AugUp_mixup_Logistic_and_Backdoor.py --dataset="HateSpeech" --transform="Sentence-BERT" --sample=$sample --alphaAug=4 --gpu=1 &
# python run_AugUp_mixup_Logistic_and_Backdoor.py --dataset="SHAC" --transform="Sentence-BERT" --sample=$sample --alphaAug=4 --gpu=2 

## Integrated mixup, ReSample, noau, into run_AugUp_Logistic_and_Backdoor.py
# sample='AlphaTrain_10-Cz_0.5'

# python run_AugUp_ReverseTranslate_Logistic_and_Backdoor.py --dataset="CD" --transform="Sentence-BERT" --sample=$sample --gpu=0 &
# python run_AugUp_ReverseTranslate_Logistic_and_Backdoor.py --dataset="HateSpeech" --transform="Sentence-BERT" --sample=$sample --gpu=1 &
# python run_AugUp_ReverseTranslate_Logistic_and_Backdoor.py --dataset="SHAC" --transform="Sentence-BERT" --sample=$sample --gpu=2 &




# python run_AugUp_Logistic_and_Backdoor.py --dataset="CD" --transform="Sentence-BERT" --augMethod="mixup" --alphaAug=4 --sample=$sample --gpu=0 &
# python run_AugUp_Logistic_and_Backdoor.py --dataset="HateSpeech" --transform="Sentence-BERT" --augMethod="mixup" --alphaAug=4 --sample=$sample --gpu=1 &
# python run_AugUp_Logistic_and_Backdoor.py --dataset="SHAC" --transform="Sentence-BERT" --augMethod="mixup" --alphaAug=4 --sample=$sample --gpu=2


# python run_AugUp_Logistic_and_Backdoor.py --dataset="CD" --transform="Sentence-BERT" --augMethod="ReSample" --sample=$sample --gpu=0 &
# python run_AugUp_Logistic_and_Backdoor.py --dataset="HateSpeech" --transform="Sentence-BERT" --augMethod="ReSample" --sample=$sample --gpu=1 &
# python run_AugUp_Logistic_and_Backdoor.py --dataset="SHAC" --transform="Sentence-BERT" --augMethod="ReSample" --sample=$sample --gpu=2 &


# python run_AugUp_Logistic_and_Backdoor.py --dataset="CD" --transform="Sentence-BERT" --augMethod="noaug" --sample=$sample --gpu=0 &
# python run_AugUp_Logistic_and_Backdoor.py --dataset="HateSpeech" --transform="Sentence-BERT" --augMethod="noaug" --sample=$sample --gpu=1 &
# python run_AugUp_Logistic_and_Backdoor.py --dataset="SHAC" --transform="Sentence-BERT" --augMethod="noaug" --sample=$sample --gpu=2



sample='AlphaTrain_5-Cz_0.5'

# idx=566
# idxSHAC=1152

# idx=6621
# idxSHAC=11063

idx=3636
idxSHAC=6114


# transform='Sentence-BERT'
transform='binaryUnigram'

# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="CD" --CombinationIdx=$idx --transform=$transform --augMethod="mixup" --alphaAug=4 --sample=$sample --gpu=0 &
# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="HateSpeech" --CombinationIdx=$idx --transform=$transform --augMethod="mixup" --alphaAug=4 --sample=$sample --gpu=1 &
# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="SHAC" --CombinationIdx=$idxSHAC --transform=$transform --augMethod="mixup" --alphaAug=4 --sample=$sample --gpu=2 &


# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="CD" --CombinationIdx=$idx --transform=$transform --augMethod="ReSample" --sample=$sample --gpu=1 &
# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="HateSpeech" --CombinationIdx=$idx --transform=$transform --augMethod="ReSample" --sample=$sample --gpu=1 &
# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="SHAC" --CombinationIdx=$idxSHAC --transform=$transform --augMethod="ReSample" --sample=$sample --gpu=2 &

# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="CD" --CombinationIdx=$idx --transform=$transform --augMethod="noaug" --sample=$sample --gpu=2 &
# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="HateSpeech" --CombinationIdx=$idx --transform=$transform --augMethod="noaug" --sample=$sample --gpu=1 &
# python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="SHAC" --CombinationIdx=$idxSHAC --transform=$transform --augMethod="noaug" --sample=$sample --gpu=2


python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="CD" --CombinationIdx=566 --transform=$transform --augMethod="LLM_Generate" --sample=$sample --gpu=2
python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="CD" --CombinationIdx=3636 --transform=$transform --augMethod="LLM_Generate" --sample=$sample --gpu=2
python run_AugUp_Logistic_and_Backdoor-FixedTraining.py --dataset="CD" --CombinationIdx=6621 --transform=$transform --augMethod="LLM_Generate" --sample=$sample --gpu=2

