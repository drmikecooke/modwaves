import numpy as np

def pack(Ej,Bj,Ub,Q,R): 
    '''Bind Ej,Bj,Ub,Q,R into one vector A'''
    return np.array([*Ej,*Bj,Ub,Q,R])

def unpack(A): 
    '''Split A into Ej,Bj,Ub,Q,R'''
    N=(len(A)-3)//2
    return A[:N],A[N:2*N],*A[2*N:]

# Reduced forms:

def pack1(Ej,Q,R): 
    '''Bind Ej,Q,R into one vector A'''
    return np.array([*Ej,Q,R])

def unpack1(A): 
    '''Split A into Ej,Q,R'''
    N=len(A)-2
    return A[:N],*A[N:]