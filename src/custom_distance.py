import math

def oneKL(a,b):
    return a * math.log(a/b)

def KL(distrA, distrB):
    # D_KL(P||Q): D_KL(A||B)
    ret = 0
    for a,b in zip(distrA, distrB):
        ret += oneKL(a,b)
    return ret


def conditionKL(distrListA, distrListB, conditionListZ):
    ret = 0
    for distrA, distrB, condZ in zip(distrListA, distrListB, conditionListZ):
        
        ret_inner = KL(distrA,distrB)
        ret_condition = ret_inner * condZ
        
        ret += ret_condition
        
    return ret