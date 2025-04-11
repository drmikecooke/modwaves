import matplotlib.pyplot as plt
import numpy as np

from .fourier import IEj

def eta(X,d,sol,lab):
    Ej=sol.x[:-2]
    if not sol.success:
        print(lab,'fail')
    plt.plot(X,IEj([d,*Ej],X),label=lab)