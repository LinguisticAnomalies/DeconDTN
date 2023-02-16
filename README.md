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
