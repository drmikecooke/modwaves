import numpy as np

def Hmx_d(L_d): # Fenton empirical formula
    return (0.141063*L_d+0.0095721*L_d**2+0.0077829*L_d**3)/(1+0.0788340*L_d+0.0317567*L_d**2+0.0093407*L_d**3)

Hmx_0=0.1410633

def d_L(L_d):
    return 0 if L_d==0 else 2*np.pi/L_d