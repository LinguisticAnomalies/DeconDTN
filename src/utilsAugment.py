import math
import numpy as np
import random


def augToN(xIn, augN, alpha, seed=17):

    # idx_list = list(combinations(range(len(xIn)),2))

    assert math.comb(len(xIn), 2) >= augN  # what if this fails?

    random.seed(seed)

    idx_list = []
    i = 0
    ret = []
    while i < augN:
        idx = tuple(np.random.permutation(len(xIn))[:2])

        if idx not in idx_list:
            idx_list.append(idx)
        else:
            continue

        _lambda = np.random.beta(alpha, alpha)

        _generated_x = _lambda * xIn[idx[0], :] + (1 - _lambda) * xIn[idx[1], :]
        # _generated_label = (_lambda*labels[idx[0]] + (1-_lambda)*labels[idx[1]]).unsqueeze(0)

        ret.append(_generated_x)
        i += 1

    return ret
