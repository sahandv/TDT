#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:30:13 2021

@author: sahand
"""

import pandas as pd
from tqdm import tqdm
# datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
datapath = '/home/sahand/GoogleDrive/Data/'

# Cora
idx_address = datapath+"Corpus/cora-classify/cora/clean/single_component_small_18k/node_idx_seq cocite"
cluster_address = datapath+"Corpus/cora-classify/cora/embeddings/single_component_small_18k/doc2vec 300D dm=1 window=10 clustering predictions"
citations_address = datapath+'Corpus/cora-classify/cora/citations'
citations_filter_address =  datapath+"Corpus/cora-classify/cora/clean/single_component_small_18k/corpus_idx_original"

# Scopus
idx_address = datapath+"Corpus/Scopus new/embeddings/doc2vec 100D dm=1 window=12 gensim41 with_id"
cluster_address = datapath+"Corpus/Scopus new/embeddings/doc2vec 100D dm=1 window=12 gensim41 with_id clustering predictions fold_9"
citations_address = datapath+'Corpus/Scopus new/clean/citations with abstracts'
# citations_filter_address =  datapath+"Corpus/cora-classify/cora/clean/single_component_small_18k/corpus_idx_original"


idx = pd.read_csv(idx_address)[['referring_id']]
clusters = pd.read_csv(cluster_address)
# citation_pairs = pd.read_csv(citations_address,sep='\t',names=['a','b'])
citation_pairs = pd.read_csv(citations_address).drop('idx',axis=1)

#filter -- optional
citations_filter =  pd.read_csv(citations_filter_address).values[:,0].tolist()
citation_pairs = citation_pairs[(citation_pairs['a'].isin(citations_filter)) | (citation_pairs['b'].isin(citations_filter))] # mask


citation_pairs = citation_pairs.values.tolist()

idx_clusters = idx.copy()
idx_clusters['cluster'] = clusters['labels']
cluster_ids = list(clusters.groupby('labels').groups.keys())
super_id = 100000000  # A number out of document ID ranges

# Connect the supernode to all cluster members
for cluster in tqdm(cluster_ids):
    cluster_papers = idx_clusters[idx_clusters['cluster']==cluster]
    # cluster_papers['cluster'] = cluster_papers['cluster'].astype(str)
    # cluster_papers = cluster_papers.values
    cluster_papers['cluster'] = ['super_'+str(i+super_id) for i in cluster_papers['cluster']]# ID for supernodes which wouldn't conflict with document IDs
    cluster_papers = cluster_papers.values.tolist()
    citation_pairs = citation_pairs+cluster_papers
    

citation_pairs = pd.DataFrame(citation_pairs,columns=['referring_id','cited_id'])#,'count'])
citation_pairs.to_csv(citations_address+' supernodes_str_name',index=False)
