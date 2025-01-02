
# python run_iw.py --dataset="HateSpeech" --crossfitFolds=0 --constraintCy --transform="binaryUnigram"
# python run_iw.py --dataset="HateSpeech" --crossfitFolds=5 --constraintCy --transform="Sentence-BERT"
# python run_iw.py --dataset="HateSpeech" --crossfitFolds=0 --constraintCy --transform="Sentence-BERT"

# python run_iw.py --dataset="CD" --crossfitFolds=0 --constraintCy --transform="binaryUnigram"
python run_iw.py --dataset="CD" --crossfitFolds=5 --transform="Sentence-BERT"
# python run_iw.py --dataset="CD" --crossfitFolds=0 --constraintCy --transform="Sentence-BERT"
python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=566 --crossfitFolds=5 --transform="Sentence-BERT"
python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=6621 --crossfitFolds=5 --transform="Sentence-BERT"

python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=566 --crossfitFolds=5 --transform="binaryUnigram"
python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=6621 --crossfitFolds=5 --transform="binaryUnigram"

python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=566 --crossfitFolds=0 --transform="Sentence-BERT"
python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=6621 --crossfitFolds=0 --transform="Sentence-BERT"


python run_iwZ_DoubleML_FixTraining.py --dataset="HateSpeech" -c=566 --crossfitFolds=5 --transform="Sentence-BERT"
python run_iwZ_DoubleML_FixTraining.py --dataset="HateSpeech" -c=6621 --crossfitFolds=5 --transform="Sentence-BERT"


python run_iwZ_DoubleML_FixTraining.py --dataset="HateSpeech" -c=566 --crossfitFolds=5 --transform="binaryUnigram"
# python run_iw.py --dataset="SHAC" --crossfitFolds=0 --constraintCy --transform="Sentence-BERT"
# python run_iw.py --dataset="SHAC" --crossfitFolds=5 --constraintCy --transform="Sentence-BERT"


# python run_iw.py --dataset="HateSpeech" --crossfitFolds=5 --constraintCy --transform="Sentence-BERT" --clf="SVM"
# python run_iw.py --dataset="HateSpeech" --crossfitFolds=0 --constraintCy --transform="Sentence-BERT" --clf="SVM"

# python run_iw.py --dataset="SHAC" --crossfitFolds=5 --constraintCy --transform="Sentence-BERT" --clf="SVM"
# python run_iw.py --dataset="SHAC" --crossfitFolds=0 --constraintCy --transform="Sentence-BERT" --clf="SVM"
python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=566 --crossfitFolds=5 --transform="Sentence-BERT" --clf="SVM"
python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=6621 --crossfitFolds=5 --transform="Sentence-BERT" --clf="SVM"
python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=3636 --crossfitFolds=5 --transform="Sentence-BERT" --clf="SVM"


python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=566 --crossfitFolds=0 --transform="Sentence-BERT" --clf="SVM"
python run_iwZ_DoubleML_FixTraining.py --dataset="CD" -c=6621 --crossfitFolds=0 --transform="Sentence-BERT" --clf="SVM"


python run_iwZ_DoubleML_FixTraining.py --dataset="SHAC" -c=1152 --crossfitFolds=5 --transform="Sentence-BERT" --clf="SVM"
python run_iwZ_DoubleML_FixTraining.py --dataset="SHAC" -c=11063 --crossfitFolds=5 --transform="Sentence-BERT" --clf="SVM"
python run_iwZ_DoubleML_FixTraining.py --dataset="SHAC" -c=6114 --crossfitFolds=5 --transform="Sentence-BERT" --clf="SVM"
python run_iwZ_DoubleML_FixTraining.py --dataset="SHAC" -c=1152 --crossfitFolds=0 --transform="Sentence-BERT"
python run_iwZ_DoubleML_FixTraining.py --dataset="SHAC" -c=6114 --crossfitFolds=0 --transform="Sentence-BERT"

### Testing...
