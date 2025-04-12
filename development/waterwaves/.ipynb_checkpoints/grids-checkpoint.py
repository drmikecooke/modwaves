import numpy as np

class grids:
    '''Set up collocation grids, 
    N: number of Fourier terms
    dscl: scaling of B coefficients,
            Fenton uses d,
            modified forms such as d+H can improve stability for high waves'''
    def __init__(self,N,dscl):
        self.Xm=np.linspace(0,np.pi,N+1) # N+1 grid for collocation, index m
        self.j=np.arange(1,N+1) # Fourier index j from 1 to N
        # create matrix of X-grid multiplied by Fourier index
        self.Xmj=self.Xm.reshape((N+1,1))@self.j.reshape((1,N))
        # evaluate cos and sin on Xmj
        self.CXmj,self.SXmj=np.cos(self.Xmj),np.sin(self.Xmj)
        self.scl=np.cosh(self.j*dscl) # scaling denominator