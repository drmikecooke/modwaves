import numpy as np
from scipy.optimize import root

from .start import linear
from .grids import grids
from .boundary import F
from .parameters import Hmx_d,Hmx_0,d_L

def wave(N,H,L_d=0,scaler=0,A0=None):
    '''Drive function giving N Fourier components and Q,R for wave of height H
    Units: k,g both equal 1, so wavelength L is 2pi
    L_d=0 is deep wave, max height Hmx_0
    scaler and A0 allow adjustments to Bj scale
    and start off point for root-finder, respectively
    Result in scipy OptimizeResult format'''
    d=d_L(L_d)
    if not np.any(A0):
        A0=linear(N,H,L_d) # use linear approximation to start root finding
    Xm=np.linspace(0,np.pi,N+1) # N+1 grid for collocation
    j=np.arange(1,N+1) # Fourier index from 1 to N
    g=grids(N,d+H) if scaler==0 else grids(N,scaler)
    return root(F,A0,args=(H,d,g)) # deliver root of F which encodes boundary conditions, and Fenton eta Fourier components (E)
    
def wave_base(N,H,L_d,fs=lambda h:None): 
    '''Create basis for interpolation, etc
    Returns list of waves for an array H of heights
    fs is a function of h=H/Hmx giving A0 for wave'''
    d=d_L(L_d)
    Hmx=Hmx_0 if L_d==0 else d*Hmx_d(L_d)
    SOL=[wave(N,h*Hmx,L_d=L_d,A0=fs(h)) for h in H]
    return SOL