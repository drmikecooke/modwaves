import numpy as np

def get(L_d,n):
    '''Read Fenton result n'''
    with open(f"fenton/data/L_d{L_d:02d}/{n:02d}.txt","rt") as fenton:
        comp=fenton.read()
    
    comp=comp.split('\n')
    
    targets=['d','H','c','Q','R','q','r']
    data={}
    jBE=[]
    for line in comp:
        if line:
            for target in targets:
                if f'({target})' in line:
                    data[target]=float(line.split('\t')[1]) #use first column of table in fenton file (L=2pi)
            if '0'<=line[1]<='9':
                jBE+=[line.split('\t')]
            
    data['Ub']=data.pop('c') # Ub is wave speed in my code
    if not 'd' in data:
        data['d']=0 # deep water
    J,B,E=np.array(jBE).T
    return data,J.astype(int),B.astype(float),E.astype(float)