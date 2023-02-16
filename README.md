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

with the constraints: <a id="binary_contraint"></a>

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


### Multi-level Confounder (Z) and Multi-level Outcome (Y)

First we start by examing constraints as in the binary setting, as in [binary-constraints](#binary_contraint). In the multi-level setting, this $\alpha_{test}$ becomes nontrivial to consolidate into a scalar value (this could be done as a vector where we compare everything to one). Thus, it could be better and simpler to just provide the full distribution before sampling:
```math
\begin{align}

& P_{train}(Y|Z) \\
& P_{test}(Y|Z) \\
& P(Z)

\end{align}
```

**Core function**: [`confoundSplitDFMultiLevel()`](src/utils.py)

Load it as module. Currently, it works with only **TWO** data sources and **BINARY** outcome.

- Input: Args:
```
df (pd.DataFrame): DataFrame, containing ALL data
z_Categories (list): unique list for confounder (z) values
y_Categories (list): unique list for outcome (y) values
z_column (str): name of confounder column
y_column (str): name of outcome column
p_train_y_given_z (list or 2D-array): P_train(Y|Z): ordered as 2D-array/nested list, where each row represents one event of Z. NOTE: row sum == 1
p_test_y_given_z (list or 2D-array): P_test(Y|Z). Same as p_train_y_given_z
p_z (list or 1D-array): distribution of confounding variable z
n_test (int, optional): number of examples in the testing set. Defaults to 100.
n_error (int, optional): number for error (+/-) when there is no exact matching for n_test. Defaults to 0.
train_test_ratio (int, optional): train:test number ratio. Defaults to 4.
seed (int, optional): random seed for train_test_split. Defaults to 2671.
```
- Output:  df_collect: list of dictionaries, each of which is for one combination of y and z. For each:

```python
{"df_train":  # df_train
 "df_test":   # df_test 
 "y":         # distinct y values 
 "z":         # distinct z values
}
```

For clarification of `p_train_y_given_z` (and similar for `p_test_y_given_z`): refer to the following table:

|   |                       | Y    |        |           |
|---|-----------------------|------|--------|-----------|
|   |                       | none | benign | malignant |
| Z | Hospital A (P(z)=0.2) | 0.1  | 0.1    | 0.8       |
|   | Hospital B (P(z)=0.2) | 0.2  | 0      | 0.8       |
|   | Hospital C (P(z)=0.3) | 0.7  | 0.3    | 0         |
|   | Hospital D (P(z)=0.3) | 0.3  | 0.3    | 0.4       |

This table corresponds to:
```python
p_train_y_given_z = [[0.1, 0.1, 0.8], 
                     [0.2, 0,   0.8], 
                     [0.7, 0.3, 0  ],
                     [0.3, 0.3, 0.4]
                    ]

p_z = [0.2, 0.2, 0.3, 0.3]
```

NOTE: the order of `y_Categories` and `z_Categories` must match with the corresponding orders in desired probability distributions. So in this case:
```python
z_Categories = ['Hosp_A','Hosp_B','Hosp_C', 'Hosp_D']
y_Categories = ['none','benign', 'malignant']

```


### Multiple Data Sources and Multi-class Outcome
**TODO**

## Model Architectures

### `NeuralModel`
not very suitable in the setting of single label prediction. Old model, originally designed for multi-class multi-label prediction, using BCE loss.

### `NeuralSingleLabelModel`
Multi-class single-label prediction framework. One prediction head. Base BERT structure.

### `GradientReverseModel`
Multi-class single-label framework, with two predictions heads: main and secondary. For example, main could be Dementia ~ No Dementia, secondary could be Pitts ~ WLS.

TODO: apply gradient reversal method!