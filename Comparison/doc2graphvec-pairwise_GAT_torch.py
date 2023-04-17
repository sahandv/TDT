#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  15 12:21:41 2022

@author: github.com/sahandv
"""

# import os
import gc
import random
# import json
import numpy as np
import pandas as pd
import networkx as nx
# from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
# from sciosci.assets import text_assets as ta
from sciosci.assets import ann_assets_torch as annat
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes
tqdm.pandas()

experiment_id = 'dim-019-12-lr=adam-100D-noshuffle'
filter_refs = False
shuffle = False
preprocess = 'numpy'
split_rate = 0.8 #train ratio

GCN_results = []
GAT_results = []

# =============================================================================
#%% Prepare data - Scopus
# =============================================================================
refs_path = '/home/sahand/Documents/scopus_refs/REF'
datapath = '/home/sahand/GoogleDrive/Data/' #Ryzen
doc_vecs = pd.read_csv(datapath+'Corpus/Scopus new/embeddings/spaced 50D dm=1 mc3 window=12 gensim41 with_id')#Corpus/Scopus new/embeddings/doc2vec 300D dm=1 window=10 b3 gensim41
# corpus_data_ = pd.read_csv(datapath+'Corpus/Scopus new/clean/abstract_title method_b_3') # get the pre-processed abstracts
corpus_data = pd.read_csv(datapath+'Corpus/Scopus new/clean/abstract_title method_b_3 with refs limited masked string') # get the pre-processed abstracts:  limited masked

ids = corpus_data['ref-ids'].values.tolist()
eids= corpus_data['ref-eids'].values.tolist()
filtered_eids= corpus_data['refs-filtered'].values.tolist()

ids_new = []
for x in tqdm(ids):
    try:
        ids_new.append([int(i) for i in x.split('|')])
    except:
        ids_new.append([])

eids_new = []
for x in tqdm(eids):
    try:
        eids_new.append(x.split('|'))
    except:
        eids_new.append([])
        
filtered_eids_new = []
for x in tqdm(filtered_eids):
    try:
        filtered_eids_new.append(x.split('|'))
    except:
        filtered_eids_new.append([])
        
corpus_data['ref-eids'] = eids_new
corpus_data['ref-ids'] = ids_new
corpus_data['refs-filtered'] = filtered_eids_new

del eids_new
del ids_new
del filtered_eids_new
del filtered_eids
del eids
del ids
gc.collect()

papers = corpus_data[['id','file']]
papers = papers.merge(doc_vecs,on='id',how='left')
papers = papers.drop_duplicates(subset=['id'])
id_dictionary = {row['file']:row['id'] for i,row in papers[['id','file']].iterrows()}
assert papers['file'].unique().shape[0]==papers.shape[0], "Dimension mismatch: There are repeated IDs or rows in data. Please fix before moving on."

# =============================================================================
# make the graph
# =============================================================================
G = nx.DiGraph()
for i,row in tqdm(corpus_data.iterrows(),total=corpus_data.shape[0]):
    for ref in row['refs-filtered']:
        G.add_edge(row['file'],ref)

# =============================================================================
# Format and organise data for NN
# =============================================================================
edges = pd.DataFrame(list(np.array(e) for e in G.edges),columns=['from','to'])
edges_ = edges.copy()
edges["from"] = edges["from"].map(lambda x: id_dictionary.get(x,x))
edges["to"] = edges["to"].map(lambda x: id_dictionary.get(x,x))

# Optional: remove citations not having an abstract. / limit the graph data to the ones in the dictionary (the ones with meta data)
edges = edges[edges["from"].isin(list(id_dictionary.values()))]
edges = edges[edges["to"].isin(list(id_dictionary.values()))]


# Obtain random indices
random_indices = np.random.permutation(range(papers.shape[0]))

# 50/50 split
train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
test_data = papers.iloc[random_indices[len(random_indices) // 2 :]]

# =============================================================================
#%% Prepare data -Cora 
# =============================================================================

datapath = '/home/sahand/GoogleDrive/Data/'
data_dir =  datapath+'Corpus/cora-classify/cora/'
label_address =  data_dir+'embeddings/with citations/corpus classes1'
edges = pd.read_csv(data_dir+'citations.withabstracts')
edges_undirected = edges.copy()
edges_undirected['referring_id'] = edges['cited_id']
edges_undirected['cited_id'] = edges['referring_id']
edges_undirected = edges_undirected.append(edges)
edges_undirected = edges_undirected.drop_duplicates()
labels = pd.read_csv(label_address,names=['class1'])
num_classes = len(labels.groupby('class1').groups.keys())
population_classes =  [len(dict(labels.groupby('class1').groups)[x]) for x in dict(labels.groupby('class1').groups)]
# read doc vecs
node_features = pd.read_csv(data_dir+'embeddings/with citations/doc2vec 100D dm=1 mc3 window=12 gensim41',index_col='id')
node_features.reset_index(inplace=True)
node_features['label'] = labels
node_features = node_features.drop_duplicates(subset=['id'])
node_ids = list(node_features.id)
node_ids = {x:i for i,x in enumerate(node_ids)}
edges_undirected.replace({"referring_id":node_ids},inplace=True)
edges_undirected.replace({"cited_id":node_ids},inplace=True)
node_labels = node_features.label
le = preprocessing.LabelEncoder()
le.fit(node_labels.values)
node_labels = le.transform(node_labels.values)
node_features.drop(['id','label'],axis=1,inplace=True)
node_features = node_features.values
edges_undirected = edges_undirected.values.T
del labels
del edges

train_mask,validation_mask,test_mask = annat.mask_maker(node_features.shape[0],train_split=0.8,val_split=0.15)
data = Data(
    x=torch.tensor(node_features, dtype=torch.float),
    edge_index=torch.tensor(edges_undirected,dtype=torch.long),
    y=torch.tensor(node_labels,dtype=torch.uint8),
    train_mask=torch.tensor(train_mask,dtype=torch.bool),
    val_mask=torch.tensor(validation_mask,dtype=torch.bool),
    test_mask=torch.tensor(test_mask,dtype=torch.bool)
    )

for key, item in data:
    print(f'{key} found in data')
print(data.num_nodes,
data.num_edges,
data.num_node_features,
data.has_isolated_nodes(),
data.has_self_loops(),
data.is_directed())

device = torch.device('cuda')
device_cpu = torch.device('cpu')

data = data.to(device)

isolated = (remove_isolated_nodes(data['edge_index'])[2] == False).sum(dim=0).item()
print(f'Number of isolated nodes = {isolated}')

print(data)


# =============================================================================
#%% Prepare data - Citeseer/cora - Online data
# =============================================================================

# Import dataset from PyTorch Geometric
dataset = Planetoid(root=".", name="Cora")

data = dataset[0]

# Print information about the dataset
print(f'Dataset: {dataset}')
print('-------------------')
# print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Print information about the graph
print(f'\nGraph:')
print('------')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')


#%% Run Classifiers


# Create GCN model
gcn = annat.GCN(data.num_features, 512, torch.max(data.y).item() + 1).to(device)
print(gcn)

# Train
annat.train_td(gcn, data)
# Test
acc,f1,precision = annat.test_td(gcn, data)
# print(f'\nGCN test accuracy: {acc*100:.2f}%\n')
print(acc,f1,precision)
GCN_results.append([acc,f1,precision])
# =============================================================================
# 
# =============================================================================

# Create GAT model
gat = annat.GAT(data.num_features, 32, torch.max(data.y).item() + 1).to(device)
print(gat)

# Train
annat.train_td(gat, data)

# Test
acc,f1,precision = annat.test_td(gat, data)
print(acc,f1,precision)
GAT_results.append([acc,f1,precision])


GCN_results_avg = np.mean(np.array(GCN_results),axis=0)
GAT_results_avg = np.mean(np.array(GAT_results),axis=0)


untrained_gat = annat.GAT(data.num_features, 32, torch.max(data.y).item() + 1).to(device)
h, _ = untrained_gat(data.x, data.edge_index)
out = h.detach()
out = out.to(device_cpu)































