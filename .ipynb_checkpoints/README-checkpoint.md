# DeconDTN

## Split Data Sets Function

### Basic Idea
Refer to the papers:

- Landeiro V, Culotta A. Robust text classification under confounding shift. Journal of Artificial Intelligence Research. 2018 Nov 5;63:391-419. ([link](https://www.jair.org/index.php/jair/article/view/11248))

- Landeiro V, Culotta A. Robust text classification in the presence of confounding bias. InThirtieth AAAI Conference on Artificial Intelligence 2016 Feb 21. ([link](https://ojs.aaai.org/index.php/AAAI/article/view/9997))


From the 2018 paper, we could have the following table:
<a id="contigency_table"></a>
|     |   | Train | Test |
|-----|---|-------|------|
| df0 | Y | a     | b    |
|     | N | c     | d    |
| df1 | Y | e     | f    |
|     | N | g     | h    |

with the constraints:

```math
\begin{align}
& p_{train}(y=1|z=0)\\

& p_{test}(y=1|z=0)\\


& p_{train}(y=1|z=1) = b_{train} \\
& p_{test}(y=1|z=1) = b_{test} \\
& p_{train}(y=1) = p_{test}(y=1) = Const_y \\
& p_{train}(z=1) = p_{test}(z=1) = Const_z

\end{align}
```

In our case (and the following code to implement this), we made some tweaks by introduing a new variable $\alpha_{test}$ (defined below), so that **given**:
<a id="factors"></a>
```math
\begin{align}
& \alpha_{test} = \frac{p_{test}(y=1|z=1)}{p_{test}(y=1|z=0)}\\

& p_{train}(y=1|z=0) = p\_pos\_train\_z0\\

& p_{train}(y=1|z=1) = p\_pos\_train\_z1\\

& p_{train}(z=1) = p_{test}(z=1) = p\_mix\_z1
\end{align}
```

we **could calculate**:
```math
\begin{align}

& p_{test}(y=1|z=0) \\
& p_{test}(y=1|z=1) \\
& p_{train}(y=1) = p_{test}(y=1) \\

\end{align}
```

Given two data sets, we need an additional parameter `train_test_ratio`. Together with [provided parameters](#factors), we call them distribution controlling parameters. Given those, the [full table](#contigency_table) could be calculated. 

### Two Data Sources and Binary Outcome

**Core function**: [`confoundSplitDF()`](src/utils.py). 

Load it as module. Currently, it works with only **TWO** data sources and **BINARY** outcome.

- Input: two dataframes, outcome column, distribution controlling parameters, random state, number of test examples (`n_test`), and error term for number of tests (`n_test_error`, meaning within range of `n_test` +/- `n_test_error`, in case the exact match is rare.)
- Output: dictionary with the following keys:

```python
{
"sample_df0_train":  # sampled df0 for train,
"sample_df0_test":   # sampled df0 for test,
"sample_df1_train":  # sampled df1 for train,
"sample_df1_test":   # sampled df1 for test,
"stats": ret         # distribution controlling parameters
}
```

The basic use case example is:
```python
 ret = confoundSplitDF(
    df0=df_wls_merge, df1=df_adress, 
    df0_label='label', df1_label='label',
    p_pos_train_z0 = 0.1, 
    p_pos_train_z1 =  0.5, 
    p_mix_z1 =  0.3, 
    alpha_test =  3,
    train_test_ratio = 4,
    random_state = 187,
    n_test = 150,
    n_test_error = 0
)
```

### Multiple Data Sources and Multi-class Outcome
**TODO**

## Model Architectures

### 1. Base BERT (`AutoModelForSequenceClassification`)
Model `NeuralSingleLabelModel()` from file [`NeuralSingleLabelModel.py`](src/NeuralSingleLabelModel.py). 

Multi-class single-label prediction framework. One prediction head. Base BERT structure.



### 2. Adversarial Model: Auxiliary Task Model (no gradient reverse)
Model `GradientReverseModel()` from file [`AdversarialModel.py`](src/AdversarialModel.py)

Multi-class single-label framework, with two predictions heads: main and secondary. For example, main could be Dementia ~ No Dementia, secondary could be Pitts ~ WLS.



### 3. Adversarial Model: Auxiliary Task Model (Gradient Reverse)
Model `GradientReverseModel()` from file [`AdversarialModel.py`](src/AdversarialModel.py)

By adding a Gradient Reversal Layer between Feature layer and **domain** classifier layer, gradients from the **domain** classifier are then reversed (negated, i.e., $grad_{domain} = grad_{domain} \times (-1)$ ). But the gradients from the **main** classifier remains the same.

Refer to the paper [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818).

Code snippets credit to [this Pytorch discussion](https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589).





### (deprecated) `NeuralModel` 
not very suitable in the setting of single label prediction. Old model, originally designed for multi-class multi-label prediction, using BCE loss.


# Navigation on This Repo

## Preprocessing

Preprocessing of each dataset is a separate file in the `src` folder:

- [`process_CD.py`](src/process_CD.py) for the Cognitive Distortion dataset. Entry point: `load_cd()`;
- [`process_HateSpeech.py`](src/process_HateSpeech.py) for the HateSpeech dataset. Entry point: `load_HateSpeech_dynGen()` and `load_HateSpeech_wsf()`;
- [`process_SHAC.py`](src/process_SHAC.py) for the SHAC. Entry point: `load_process_SHAC()`.

Use the entry points for detailed preprocessing steps for each dataset.

## General Pipeline

Model training (or adjustment) - Inference - Evaluation

All the documented tasks follow this pipeline, with some little difference within some step(s).

Tasks include Task Vector, Backdoor Adjustment, DistMatch framework, Augmentation, MMD.

## TaskVector

The typical workflow consists of 3 major steps: fine-tuning model and editing weights; inference; evaluation. 


The following takes RoBERTa model as an example (in the file [`runs_xiruo/run_roberta_EndToEnd_1.0exp.sh`](runs_xiruo/run_roberta_EndToEnd_1.0exp.sh)):

- Step 1.1: fine-tuning the model.
    - Predict Target (Label): `python BERT_finetuning_freezeQV.py --CombinationIdx=3636 --model_name='roberta-base' --dataset=$dt --toPredict='Target'  --gpu='3'`
    - Predict Provenance (Reverse Label): `python BERT_finetuning_freezeQV.py --CombinationIdx=3636 --model_name='roberta-base' --dataset=$dt --toPredict='Source' --reverseLabel  --gpu='3'`
    - NOTE: `BERT_finetuning_freezeQV.py` follows a deprecated naming convention. "freezeQV" doesn't function (no layer gets frozen), so this RoBERTa model is fully trained.

- Step 1.2: editing model weights. `python BERT_editingWeights.py --model_name='roberta-base' --target_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/set-$isample-epoch3" --source_model_id="/bime-munin/xiruod/roberta-base_$dt/n200/Source-set-$isample-epoch3" --weightsEditedDir="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/" --gpu=$gpu --lambda1=$ilambda1 --lambda2=$i`

- Step 2: inference. `python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt/n200/Weights/set-$isample-epoch3-lambda1_$ilambda1-lambda2_$i-added.pth" --output_dir="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt`

- Step 3: evaluation. `python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt/n200/Inferences/inference_set-$isample-epoch3-lambda1_$ilambda1-lambda2_$i-added" --output_dir="../output/roberta/$dt/" --nRuns=5 --dataset=$dt`

`$dt`, `$gpu`, `$ilambda1`, `$i` are placeholders for shell scripting. Each of those will be replaced by its true value (details in the sh script).

## mixup

`mixup` is experimented in the file: [`runs_xiruo/run_mixup_by4.sh`](runs_xiruo/run_mixup_by4.sh)

The model training step is `python mixup_by4.py --dataset=$dt --CombinationIdx=$isample --num_train_epochs=6`

Then use Step 2 and Step 3 from TaskVector section.

## DistMatch

Refer to the file ['runs_xiruo/run_roberta_AugUp.sh'](runs_xiruo/run_roberta_AugUp.sh).


- Step 1: fine-tuning under DistMatch. `python BERT_finetuning_AugUp.py --CombinationIdx=$isample --model_name='roberta-base' --dataset=$dt --augMethod=$augm  --gpu=$gpu --num_train_epochs=$ep`
- Step 2: inference. `python BERT_edited_weights_InferenceOnly.py --model_name='roberta-base' --weightsEdited="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/set-$isample-epoch$ep/pytorch_model.bin" --output_dir="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep/" --gpu=$gpu --device="cuda:0" --batch_size=8 --dataset=$dt`
- Step 3: evaluation`python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/roberta-base_$dt-FullFT-$augm/n200/Inferences-epoch$ep/inference_set-$isample-epoch$ep" --output_dir="../output/roberta/$dt-FullFT-$augm-epoch$ep/" --nRuns=5 --dataset=$dt`

## Backdoor Adjustment

Refer to the file [`runs_xiruo/run_backdoor.sh`](runs_xiruo/run_backdoor.sh). Step 1 generates two models: one original logistic regression model and the Backdoor adjusted model.


- Step 1: train one original logistic regression model and the Backdoor adjusted model. `python Backdoor_TrainOnFix_and_Inference.py  --CombinationIdx=$isample --model_name='roberta-base' --dataset=$dt  --nTest=200 --transform='binaryUnigram' --runs=1 --output_dir="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_Backdoor/" --output_dir_noBackdoor="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_noBackdoor/"`
- Step 2: inference. `python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_Backdoor/inference_set-$isample-irun-0" --output_dir="../output/backdoor/$dt/backdoor/" --nRuns=5 --dataset=$dt`
- Step 3: evaluation. `python ../notebooks_xiruo/run_edited_weights_Eval.py --bert --inferencePathPrefix="/bime-munin/xiruod/backdoor_$dt/n200/Inferences_noBackdoor/inference_set-$isample-irun-0" --output_dir="../output/backdoor/$dt/noBackdoor/" --nRuns=5 --dataset=$dt`





