#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:47:08 2022

@author: github.com/sahandv
"""

import random
from collections import Counter
from tqdm import tqdm

def walk(G,path:list,p,q,s,sf,l):
    """
    Create a walk from path list with at least one element as the starting point

    Parameters
    ----------
    G : Networkx Graph
    path : list
        list of startings with node names
    p : float
        P parameter. Controls the probablility of returning. (high=less)
        If 0, will not return. Turning into DeepWalk.
    q : float
        Q parameter. Controls the probablility of moving forward. (high=less)
    s : float
        S parameter. Controls the probablility of revisiting. (high=less)
    sf : float
        SF parameter. Controls the rate to decrease in the probablility of multiple times revisiting (higher=less)
    l : int
        walk length.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    path_ = next_node(G,path,p,q,s,sf)
    if len(path_)<=l:
        return walk(G,path_,p,q,s,sf,l)
    else:
        return path
        
def next_node(G,path,p,q,s,sf):
    """
    Get the next node according to the hyper parameters. 
    If skip results in the removal of nodes, skip will be ignored.

    Parameters
    ----------
    G : Networkx Graph
    path : list
        list of startings with node names
    p : float
        P parameter. Controls the probablility of returning. (high=less)
        If 0, will not return. Turning into DeepWalk.
    q : float
        Q parameter. Controls the probablility of moving forward. (high=less)
    s : float
        S parameter. Controls the probablility of revisiting. (high=less)
    

    Returns
    -------
    str
        Name of the next node.

    """
    node = path[-1]
    neighbors = list(G.neighbors(node))
    counts = Counter(path)
    probabilities = [(1/q) if n not in path else (1/(q*s*max(1,sf*(counts[n]-1)))) for n in neighbors]
    try: #set the probablility for returning to the previous node
        prev_n = neighbors.index(path[-2]) 
        if p==0:
            probabilities[prev_n] = 0
        probabilities[prev_n] = 1/p
    except:
        pass
    # Randomly select one
    return path+[random.choices(neighbors,probabilities)[0]]


def sample(G,p=1.0,q=0.5,s=3,sf=1.2,l=20,n_walks=50,nodes_list=[],id_only=False):
    walks = []
    if len(nodes_list)==0:
        all_nodes = list(G.nodes())
    else:
        all_nodes = nodes_list
        
    for node in tqdm(all_nodes,total=len(all_nodes)): 
        start = [node]
        for walk_n in range(n_walks):  
            if id_only:
                walks.append([w.split('-')[-1] for w in walk(G,start,p,q,s,sf,l)])
            else:
                walks.append(walk(G,start,p,q,s,sf,l))
    return walks
