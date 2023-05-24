#!/usr/bin/env python3

# description: simple toy example demonstrating how to use the ORC class to compute curvature
# created on 1 dec 22
# @author: Jiening Zhu

import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import networkx as nx
import pandas as pd
import time
import re
import numpy as np
from GeometricNetworkAnalysis.ORC import ORC

adj=pd.read_csv("./data/adj.txt",header=None)
hprd_genes=pd.read_csv("./data/hprd.txt",header=None)
hprd_genes=hprd_genes.values.tolist()
hprd_genes=[k[0] for k in hprd_genes]
adj.columns=hprd_genes
adj.index=hprd_genes

#input_folder="/home/jiening666/Data/mm_CoMMpass/out/"
#omics=["RNA","CNA"]
 
input_folder="./data/lgg_tcga/out/"
omics=["RNA","CNA","Methyl"]
#omics=["CNA"]
for omic in omics:
    input_name=input_folder+omic+".csv"
    tb=pd.read_csv(input_name,header=0,index_col=0)
    test=tb.index.tolist()

    start = time.time()
    # --- Setup weighted graph ---
    #G = nx.karate_club_graph()
    #G = nx.fast_gnp_random_graph(500, 0.1, seed=10)
    adj_s=[[adj[i][j] for j in test] for i in test]
    adj_s=np.array(adj_s)
    adj_s=adj_s-np.diag(np.diag(adj_s))
    G=nx.from_numpy_matrix(np.array(adj_s))
    largest_cc = max(nx.connected_components(G), key=len)
    G=G.subgraph(largest_cc).copy()
    results=[]
    start = time.time()

    if np.min(tb.values)<-10**-6:
        is_exp=True
    else:
        is_exp=False
        
    for i in range(tb.shape[1]):
        if omic=="CNA":
            nx.set_node_attributes(G, {k:np.exp(tb.values[k][i]) for k in G}, name='weight')
        elif omic=="Methyl":
            nx.set_node_attributes(G, {k:1-tb.values[k][i]+10**-6 for k in G}, name='weight')
        else:
            nx.set_node_attributes(G, {k:tb.values[k][i]+10**-6 for k in G}, name='weight')
        orc = ORC(G, verbose="ERROR")
        edge_curvatures = orc.compute_curvature_edges()
        end = time.time()
        results.append([edge_curvatures[key] for key in sorted(edge_curvatures.keys())]) 
        print(omic,i+1,"/",tb.shape[1],",t=",time.strftime("%H:%M:%S", time.gmtime(end - start)),"/",time.strftime("%H:%M:%S", time.gmtime((end - start)/(i+1)*tb.shape[1])))

    out=pd.DataFrame(results,index=tb.columns,columns=sorted(edge_curvatures.keys()))
    output_name=input_folder+omic+"_curv.csv"
    out.to_csv(output_name)