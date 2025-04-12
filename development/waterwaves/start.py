import numpy as np

from .packing import pack

def linear(N,H,L_d):
    '''Linear approximation to wave form'''
    Ej0=np.zeros(N)
    Ub0=1 if L_d==0 else np.tanh(2*np.pi/L_d)**(1/2)
    Ej0[0],d=H/2,0 if L_d==0 else 2*np.pi/L_d
    Q0,R0=Ub0*d,Ub0**2/2+d
    return pack(Ej0,Q0,R0) # pack them into vector