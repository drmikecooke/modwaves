import numpy as np

def FEtam(Etam):
    '''Convert Eta array to fourier coefficients, 0 order is 2*d'''
    N=len(Etam)-1
    Xm=np.linspace(0,np.pi,N+1)
    j=np.arange(0,N+1)
    Xmj=Xm.reshape((N+1,1))@j.reshape((1,N+1))
    return 2*(Etam[1:-1]@np.cos(Xmj)[1:-1])/N+Etam[0]*np.cos(Xmj)[0]/N+Etam[-1]*np.cos(Xmj)[-1]/N

def IEj(Ej,X):
    '''Use Ej to interpolate eta at values in X-array, Ej[0]=d'''
    N=len(Ej)
    CXj=np.cos(X.reshape(len(X),1)@np.arange(0,N).reshape(1,N))
    return CXj@Ej