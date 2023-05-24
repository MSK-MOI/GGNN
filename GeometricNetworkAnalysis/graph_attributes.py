"""
    Functions for computing graph attributes
"""

from functools import lru_cache
import networkx as nx
import numpy as np

EPS = 1e-7
cache_maxsize=1000000

@lru_cache(cache_maxsize)
def pij(G,source,target,n_weight="weight",EPS=1e-7):
    """ Compute the 1-step Markov transition probabiltiy of going from source to target node in G 
    Note: not a lazy walk (i.e. alpha=0)"""
    assert G.nodes[source][n_weight] > EPS,f"Node {source} with weight < EPS does not interact with any other nodes."
    if target not in G.neighbors(source):
        return 0.0
    w = [G.nodes[nbr][n_weight] for nbr in G.neighbors(source)]
    if sum(w) > EPS: # ensure no dividing by zero
        return G.nodes[target][n_weight]/sum(w)
    else: # ensure no dividing by zero
        logger.warning("p_ij({},{}) - using uniform distribution because weighted ndoal degree too small.".format(source,target))
        return G.nodes[target][n_weight]/len(list(G.neighbors(source)))
    
def compute_edge_weights(G : nx.Graph,n_weight="weight",e_weight="weight",e_normalized=False,e_sqrt=False):
    """ compute edge weights from given nodal weights 
    e_normalized = True AND e_sqrt = True : w_ij = 1/sqrt([(p_ij + p_ji)/2])
    e_normalized = True AND e_sqrt = False : w_ij = 1/[(p_ij + p_ji)/2]
    e_normalized = False AND e_sqrt = True : w_ij = 1/sqrt(w_i * w_j)
    e_normalized = False AND e_sqrt = False : w_ij = (1/w_i)*(1/w_j) 
    NOTE: w_ij = INF if w_i=0 or w_j=0
    """
    assert ~(not nx.get_node_attributes(G,n_weight)), "Node weight not detected in graph."
    
    # compute edge weight
    weights = {}
    if e_normalized: # normalized
        for i,j in G.edges():
            wij = pij(G,i,j,n_weight)
            wji = pij(G,j,i,n_weight)
            w = (wij+wji)/2 # d(i,j) = 1/sqrt(w_ij)
            if e_sqrt: # d(i,j) = 1/sqrt(w_ij)
                w = np.sqrt(w)
            weights[(i,j)] = 1/w if w > EPS else np.inf        
    else: # not normalized
        for i,j in G.edges():
            w = G.nodes[i][n_weight]*G.nodes[j][n_weight] # w = (1/w_i)*(1/w_j)
            if e_sqrt: # w_ij = 1/sqrt(w_i * w_j)"
                w = np.sqrt(w)
            weights[(i,j)] = 1/w if w > EPS else np.inf
    nx.set_edge_attributes(G, weights, name=e_weight)
    return G
    
