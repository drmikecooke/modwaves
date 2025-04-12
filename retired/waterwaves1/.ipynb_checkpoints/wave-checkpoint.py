import numpy as np
from scipy.optimize import root

from .start import linear,linear1
from .grids import grids
from .boundary import F,F1

def wave1(N,H,d=0,scaler=0,A0=None): # d=0 default for deep water, scaler default d+H
    # I use k,g=1,1 units so wavelength is 2pi
    if not np.any(A0):
        A0=linear1(N,H,d) # use linear approximation to start root finding
    Xm=np.linspace(0,np.pi,N+1) # N+1 grid for collocation
    j=np.arange(1,N+1) # Fourier index from 1 to N
    g=grids(N,d+H) if scaler==0 else grids(N,scaler)
    return root(F1,A0,args=(H,d,g)) # deliver root of F which encodes boundary conditions, and Fenton eta Fourier components (E)

def Hmx_d(L_d): # Fenton empirical formula
    return (0.141063*L_d+0.0095721*L_d**2+0.0077829*L_d**3)/(1+0.0788340*L_d+0.0317567*L_d**2+0.0093407*L_d**3)
    
def wave_base(N,H,L_d,fs=lambda h:None): # create basis for interpolation, etc
    d,Hmx=2*np.pi/L_d,Hmx_d(L_d)
    SOL=[wave1(N,h*Hmx*d,d=d,A0=fs(h)) for h in H]
    return SOL