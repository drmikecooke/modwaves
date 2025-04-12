import matplotlib.pyplot as plt
import numpy as np

from .fourier import IEj
from .wave import wave_base

def eta(X,d,sol,lab): # plot eta versus x
    Ej=sol.x[:-2]
    if not sol.success:
        print(lab,'fail')
    plt.plot(X,IEj([d,*Ej],X),label=lab)

def extract(SOL,attr):
    return [getattr(sol,attr) for sol in SOL]
    
def Ej(H,SOL): # plot Ej versus h given L/d
    SOLx=np.array(extract(SOL,'x'))    
    plt.plot(H,SOLx[:,:-2],label=[f'j={j}' for j in range(1,SOLx.shape[1]-1)])

def QR(H,SOL): # plot QR versus h given L/d
    SOLx=np.array(extract(SOL,'x'))
    plt.plot(H,SOLx[:,-2:],label=['Q','R'])