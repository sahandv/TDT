#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:27:46 2020

@author: github.com/sahandv
"""
import sys
import gc
import warnings
from text_unidecode import unidecode
from collections import deque
warnings.filterwarnings('ignore')
from scipy import spatial


import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
# conda install -n base -c conda-forge widgetsnbextension
# conda install -c conda-forge ipywidgets
from node2vec import Node2Vec
from gensim.models import Word2Vec

sns.set_style('whitegrid')
seed=1
np.random.seed(seed)
data_not_clean= False
# =============================================================================
# Read data
# =============================================================================
dir_path = '/home/sahand/GoogleDrive/Data/Corpus/cora-classify/cora/'
dir_path = '/home/sahand/GoogleDrive/Data/Corpus/Scopus new/clean/'
# dir_path = '/home/sahand/GoogleDrive/Data/Corpus/Dimensions AI unlimited citations/clean/'
# data = pd.read_csv(dir_path+'citations_filtered_single_component.csv')# with_d2v300D_supernodes.csv')#, names=['referring_id','cited_id'],sep='\t')
# data = pd.read_csv(dir_path+'citations pairs - int')# with_d2v300D_supernodes.csv')#, names=['referring_id','cited_id'],sep='\t')
data = pd.read_csv(dir_path+'citations with abstracts supernodes_9')#, names=['referring_id','cited_id'],sep='\t')# with_d2v300D_supernodes.csv')#, names=['referring_id','cited_id'],sep='\t')
# data = pd.read_csv(dir_path+'clean/single_component_small_18k/network_cocitation with_d2v300D_supernodes')#network with_d2v300D_supernodes.csv')#, names=['referring_id','cited_id'],sep='\t')

if data_not_clean:
    tmp = []
    def extract_keywords(s):
        try:
            return s.split('|')
        except :
            return []
    data = data[['id','ref-ids']]
    data['ref-ids-ar'] = data['ref-ids'].apply(lambda x: extract_keywords(x))
    
    for i,row in tqdm(data.iterrows(),total=data.shape[0]):
        cites = row['ref-ids-ar']
        for cited in cites:
            tmp.append([row['id'],cited])
    
    data = pd.DataFrame(tmp)
    del tmp
    
# data = data[['id']].astype(int).astype(str)

data.columns = ['referring_id','cited_id']#,'count']
data['referring_id'] = data['referring_id'].astype(int).astype(str)
data['cited_id'] = data['cited_id'].astype(int).astype(str)

gc.collect()
sample = data.sample()
data.info(memory_usage='deep')

# Filter
# idx = pd.read_csv(dir_path+'publication idx',names=['id'])#(dir_path+'corpus idx',index_col=0)
idx = pd.read_csv(dir_path+'embeddings/with citations/corpus idx')#(dir_path+'corpus idx',index_col=0)
idx.columns = ['id']
idx['id'] = idx['id'].str.replace('pub.','').astype(str).astype(int)
idx = idx['id'].astype(str).values.tolist()

data = data[(data['referring_id'].isin(idx)) & (data['cited_id'].isin(idx))] # mask
# data.to_csv(dir_path+'citations.withabstracts',index=False)
sample = data.sample(5000)
# =============================================================================
# Prepare graph
# =============================================================================
graph = nx.Graph()
for i,row in tqdm(data.iterrows(),total=data.shape[0]):
    graph.add_edge(row['referring_id'],row['cited_id'])

print('Graph fully connected:',nx.is_connected(graph))
print('Connected components:',nx.number_connected_components(graph))

# connected_components = list(nx.connected_components(graph))

gc.collect()
# =============================================================================
# Train
# =============================================================================
node2vec = Node2Vec(graph, dimensions=100, walk_length=5, num_walks=50, workers=16, p=2, q=1,seed=seed,quiet=True)
model = node2vec.fit(window=5, min_count=1)
# model.save('/home/sahand/GoogleDrive/Data/Corpus/Scopus new/models/n2v 100D p2 q1 len5-50')
model.save(dir_path+'../models/node2vec 100D p2 q1 len5-w50 supernode_9 19-10-2022')

# =============================================================================
# Get embeddings
# =============================================================================
model_name = 'node2vec 100D p2 q1 len5-w50 supernode_9 19-10-2022'
model = Word2Vec.load(dir_path+'models/'+model_name)
# model = Word2Vec.load('/home/sahand/GoogleDrive/Data/Corpus/Scopus new/models/n2v 100D p2 q1 len5-50')

# model_name = 'ont embedding p1 q05 len10'
# model = Word2Vec.load('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/'+model_name)

embeddings = []
idx_true = []
miss_count = 0
unique_ids = pd.DataFrame(data['referring_id'].values.tolist()+data['cited_id'].values.tolist(),columns=['id']).drop_duplicates()
for i,row in tqdm(unique_ids.iterrows(),total=unique_ids.shape[0]):
    i = row['id']
    # embeddings.append(model.wv[str(i)])
    try:
        embeddings.append(model.wv[str(i)])
        idx_true.append(i)
    except:
        miss_count+=1
        print('Error while getting the embedding',i,':',sys.exc_info()[0])
print('total misses:',miss_count)

embeddings = pd.DataFrame(embeddings)
embeddings.index = idx_true
# embeddings.to_csv('/home/sahand/GoogleDrive/Data/Corpus/Scopus new/embeddings/n2v 100D p2 q1 len5-50',index=True)
embeddings.to_csv(dir_path+'../embeddings/'+model_name,index=True)

# =============================================================================
# Get embeddings for concept mapss
# =============================================================================
model = Word2Vec.load('/home/sahand/GoogleDrive/Data/Corpus/Taxonomy/ont embedding 50D p1 q05 len20-30')
path = '/home/sahand/GoogleDrive/Data/Corpus/Scopus new/clean/'
datapath = '/home/sahand/GoogleDrive/Data/' #Ryzen

doc_concepts = pd.read_csv(datapath+'Corpus/Dimensions All/clean/kw ontology search/mapped concepts for keywords')
doc_concepts = doc_concepts.fillna('')
doc_concepts = doc_concepts['concepts'].str.split('|').values.tolist()

vecs = []
for i,doc in tqdm(enumerate(doc_concepts),total=len(doc_concepts)):
    vec = list()
    
    for concept in doc:
        try:
            vec.append(model.wv[concept])
        except :
            pass
    if len(vec)==0:
        vec = [np.zeros(50)]
        print(i,'empty data row')        
    vecs.append(np.array(vec).mean(axis=0))

pd.DataFrame(vecs).to_csv('/home/sahand/GoogleDrive/Data/Corpus/Dimensions All/embeddings/concepts node2vec 50D p1 q05 len20 average of concepts')
pd.DataFrame(vecs).to_csv('/home/sahand/GoogleDrive/Data/Corpus/Scopus new/embeddings/concepts/node2vec/50D p1 q05 len20 average of concepts')
