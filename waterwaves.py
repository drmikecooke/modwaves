import numpy as np
from scipy.optimize import root

def pack(Ej,Bj,Ub,Q,R): 
    '''Bind Ej,Bj,Ub,Q,R into one vector A'''
    N=len(Bj)
    A=np.zeros(2*N+3)
    A[:N]=Ej
    A[N:2*N]=Bj
    A[2*N:]=Ub,Q,R
    return A

def unpack(A): 
    '''Split A into Ej,Bj,Ub,Q,R'''
    N=(len(A)-3)//2
    return A[:N],A[N:2*N],*A[2*N:]

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

def linear(N,H,d):
    '''Linear approximation to wave form'''
    Bj0=np.zeros(N) # initial guess based on linear approximation
    Ej0=np.zeros(N)
    Ub0=1 if d==0 else np.tanh(d)**(1/2)
    Bj0[0]=H/2/Ub0
    Ej0[0]=H/2
    Q0,R0=Ub0*d,Ub0**2/2+d    
    return pack(Ej0,Bj0,Ub0,Q0,R0) # pack them into vector

def F(A,H,d,g):
    '''Set of conditions on boundaries'''
    Ej,Bj,Ub,Q,R=unpack(A)
    N=len(Ej)
    Etam=IEj([d,*Ej],g.Xm) # imposes depth constraint
    Etamj=Etam.reshape((N+1,1))@g.j.reshape((1,N))
    C,S=(np.exp,np.exp) if d==0 else (np.cosh,np.sinh) # hyperbolics for Y dependence, for deep water use exp
    CEmj,SEmj=C(Etamj)/g.scl,S(Etamj)/g.scl
    kin=-Ub*Etam+((SEmj*g.CXmj))@(Bj)+Q # 2N+1 kinetic conditions
    Um=-Ub+((CEmj*g.CXmj))@(g.j*Bj)
    Vm=((SEmj*g.SXmj))@(g.j*Bj)
    dyn=(Um**2+Vm**2)/2+Etam-R # 2N dynamic conditions
    height=Etam[0]-Etam[-1] # assume height is maximum-minium given by eta at x=0,pi, respectively
    return np.hstack([kin,dyn,height-H])

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

def wave(N,H,d=0,scaler=0): # d=0 default for deep water, scaler default d+H
    # I use k,g=1,1 units so wavelength is 2pi
    A0=linear(N,H,d) # use linear approximation to start root finding
    Xm=np.linspace(0,np.pi,N+1) # N+1 grid for collocation
    j=np.arange(1,N+1) # Fourier index from 1 to N
    g=grids(N,d+H) if scaler==0 else grids(N,scaler)
    return root(F,A0,args=(H,d,g)) # deliver root of F which encodes boundary conditions, and Fenton eta Fourier components (E)
