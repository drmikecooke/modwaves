import numpy as np

from .packing import unpack,unpack1
from .fourier import IEj

def F(A,H,d,g):
    '''Set of conditions on boundaries'''
    Ej,Bj,Ub,Q,R=unpack(A)
    N=len(Ej)
    Etam=IEj([d,*Ej],g.Xm) # imposes depth constraint
    Etamj=Etam.reshape((N+1,1))@g.j.reshape((1,N))
    C,S=(np.exp,np.exp) if d==0 else (np.cosh,np.sinh) # hyperbolics for Y dependence, for deep water use exp
    CEmj,SEmj=C(Etamj)/g.scl,S(Etamj)/g.scl
    kin=-Ub*Etam+((SEmj*g.CXmj))@(Bj)+Q # N+1 kinetic conditions
    Um=-Ub+((CEmj*g.CXmj))@(g.j*Bj)
    Vm=((SEmj*g.SXmj))@(g.j*Bj)
    dyn=(Um**2+Vm**2)/2+Etam-R # N+1 dynamic conditions
    height=Etam[0]-Etam[-1] # assume height is maximum-minium given by eta at x=0,pi, respectively
    return np.hstack([kin,dyn,height-H]) # 2N+3 conditions total

def CSEm_Ejg(Ej,d,g):
    N=len(Ej)
    Etam=IEj([d,*Ej],g.Xm) # imposes depth constraint
    Etamj=Etam.reshape((N+1,1))@g.j.reshape((1,N))
    C,S=(np.exp,np.exp) if d==0 else (np.cosh,np.sinh) # hyperbolics for Y dependence, for deep water use exp
    return C(Etamj)/g.scl,S(Etamj)/g.scl,Etam

def UbBj_SEQg(S,E,Q,g):
    N=len(E)-1
    a=np.hstack([-E.reshape(N+1,1),(S*g.CXmj)])
    b=-Q*np.ones(N+1)
    return np.linalg.solve(a,b)

def F1(A,H,d,g):
    '''Set of reduced conditions on boundaries'''
    Ej,Q,R=unpack1(A)
    N=len(Ej)
    CEmj,SEmj,Etam=CSEm_Ejg(Ej,d,g)
    UbBj=UbBj_SEQg(SEmj,Etam,Q,g) # impose kinetic condition to derive Ub,Bj
    Ub,Bj=np.split(UbBj,[1])
    Um=-Ub+((CEmj*g.CXmj))@(g.j*Bj)
    Vm=((SEmj*g.SXmj))@(g.j*Bj)
    dyn=(Um**2+Vm**2)/2+Etam-R # N+1 dynamic conditions
    height=Etam[0]-Etam[-1] # assume height is maximum-minium given by eta at x=0,pi, respectively
    return np.hstack([dyn,height-H]) # N+2 conditions total