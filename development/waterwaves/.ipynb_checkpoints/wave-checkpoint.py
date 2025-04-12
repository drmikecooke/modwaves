import numpy as np
from scipy.optimize import root

from .start import linear
from .grids import grids
from .boundary import F
from .parameters import Hmx_d

def wave(N,H,L_d=0,scaler=0,A0=None): # d=0 default for deep water, scaler default d+H
    # I use k,g=1,1 units so wavelength is 2pi
    if not np.any(A0):
        A0=linear(N,H,L_d) # use linear approximation to start root finding
    d=0 if L_d==0 else 2*np.pi/L_d
    Xm=np.linspace(0,np.pi,N+1) # N+1 grid for collocation
    j=np.arange(1,N+1) # Fourier index from 1 to N
    g=grids(N,d) if scaler==0 else grids(N,scaler)
    return root(F,A0,args=(H,d,g)) # deliver root of F which encodes boundary conditions, and Fenton eta Fourier components (E)
    
def wave_base(N,H,L_d,fs=lambda h:None): # create basis for interpolation, etc
    d,Hmx=2*np.pi/L_d,Hmx_d(L_d)
    SOL=[wave1(N,h*Hmx*d,d=d,A0=fs(h)) for h in H]
    return SOL