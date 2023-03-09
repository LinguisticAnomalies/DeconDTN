import math

def oneKL(a,b):
    return a * math.log(a/b)

def KL(distrA, distrB):
    # D_KL(P||Q): D_KL(A||B)
    ret = 0
    for a,b in zip(distrA, distrB):
        ret += oneKL(a,b)
    return ret