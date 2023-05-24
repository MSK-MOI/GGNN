import numpy as np
import pandas as pd
def find_1_neighbors(Adj):
    Adj=np.array(Adj)
    n=Adj.shape[0]
    Adj= Adj - np.diag(np.diag(Adj)) + np.eye(n)
    return Adj

def find_edge_neighbors(Adj):
    Adj=np.array(Adj)
    n=Adj.shape[0]
    Adj_t=np.triu(Adj, k=1)
    Adj= Adj - np.diag(np.diag(Adj)) + np.eye(n)
    K=np.nonzero(Adj_t)
    indicator=np.zeros([len(K[0]),n])
    for count in range(len(K[0])):
        a=K[0][count]
        b=K[1][count]
        v=Adj[a,:]+Adj[b,:]
        v=v>0
        indicator[count,:]=v
    indicator=indicator.astype(float)
    return indicator

def find_pathway(Path_dir,Adj,gene_list):
    T = pd.read_excel(Path_dir,header=None)
    pathway_names=T.values[:,0]
    N = T.values.shape[0]
    n=Adj.shape[0]
    Adj_t=np.triu(Adj, k=1)
    Adj= Adj - np.diag(np.diag(Adj)) + np.eye(n)
    K=np.nonzero(Adj_t)
    indicator=np.zeros([N,len(K[0])])
    for i in range(N):
        v=T.values[i,:]
        v=[item for item in v if pd.isnull(item) == False]
        for count in range(len(K[0])):
            a=K[0][count]
            b=K[1][count]
            genea=gene_list[a]
            geneb=gene_list[b]
            if (genea in v) & (geneb in v):
                indicator[i,count]=1
    v=np.sum(indicator,axis=0)
    v=1-(v>0)
    indicator=np.vstack((indicator,v))
    indicator=indicator.astype(float)
    return indicator