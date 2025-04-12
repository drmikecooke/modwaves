import numpy as np

from .packing import pack,pack1

def linear(N,H,d):
    '''Linear approximation to wave form'''
    Bj0=np.zeros(N) # initial guess based on linear approximation
    Ej0=np.zeros(N)
    Ub0=1 if d==0 else np.tanh(d)**(1/2)
    Bj0[0]=H/2/Ub0
    Ej0[0]=H/2
    Q0,R0=Ub0*d,Ub0**2/2+d
    return pack(Ej0,Bj0,Ub0,Q0,R0) # pack them into vector

def linear1(N,H,d):
    '''Linear approximation to wave form'''
    Ej0=np.zeros(N)
    Ub0=1 if d==0 else np.tanh(d)**(1/2)
    Ej0[0]=H/2
    Q0,R0=Ub0*d,Ub0**2/2+d
    return pack1(Ej0,Q0,R0) # pack them into vector