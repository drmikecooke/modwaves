import numpy as np

def pack(Ej,Q,R): 
    '''Bind Ej,Q,R into one vector A'''
    return np.array([*Ej,Q,R])

def unpack(A): 
    '''Split A into Ej,Q,R'''
    N=len(A)-2
    return A[:N],*A[N:]