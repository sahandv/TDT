#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  15 12:21:41 2022

@author: github.com/sahandv
"""

import os
import gc
import random
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import multiprocessing
# from multiprocessing.pool import ProcessPoolExecutor as Pool

import concurrent.futures
import tensorflow as tf
from tensorflow import keras 
# from tensorflow.keras.preprocessing.text import Tokenizer,WordpieceTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import Tokenizer,BertWordPieceTokenizer, models, pre_tokenizers, decoders, trainers, processors
import matplotlib.pyplot as plt
from tqdm import tqdm
from sciosci.assets import text_assets as ta
from sciosci.assets import ann_assets as anna

from sklearn.model_selection import train_test_split
# !wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'
from tensorflow.keras import mixed_precision
tqdm.pandas()

experiment_id = 'dim-019-12-lr=adam-100D-noshuffle'
filter_refs = False
shuffle = False
preprocess = 'numpy'
split_rate = 0.8 #train ratio
embedding_dim = 100
input_dim = 100
output_dim = 100
n_inputs = 1 #network type selection
num_epochs = 40
vocab_limit = 40000
batch_size = 500

# =============================================================================
# Prepare GPU
# =============================================================================
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
mixed_precision.set_global_policy('mixed_float16')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
   
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

edges = tf.convert_to_tensor(edges.astype(str).values)

# Obtain random indices
random_indices = np.random.permutation(range(papers.shape[0]))

# 50/50 split
train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
test_data = papers.iloc[random_indices[len(random_indices) // 2 :]]

# =============================================================================
#%% Prepare data - Cora
# =============================================================================

zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)

data_dir = os.path.join(os.path.dirname(zip_file), "cora")

citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)

papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"),
    sep="\t",
    header=None,
    names=["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"],
)

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

# Obtain random indices
random_indices = np.random.permutation(range(papers.shape[0]))

# 50/50 split
train_data = papers.iloc[random_indices[: len(random_indices) // 2]]
test_data = papers.iloc[random_indices[len(random_indices) // 2 :]]

#%% 

# Obtain paper indices which will be used to gather node states
# from the graph later on when training the model
train_indices = train_data["paper_id"].to_numpy()
test_indices = test_data["paper_id"].to_numpy()

# Obtain ground truth labels corresponding to each paper_id
train_labels = train_data["subject"].to_numpy()
test_labels = test_data["subject"].to_numpy()

# Define graph, namely an edge tensor and a node feature tensor
edges = tf.convert_to_tensor(citations[["target", "source"]])
node_states = tf.convert_to_tensor(papers.sort_values("paper_id").iloc[:, 1:-1])

# Print shapes of the graph
print("Edges shape:\t\t", edges.shape)
print("Node features shape:", node_states.shape)


# Define hyper-parameters
HIDDEN_UNITS = 100
NUM_HEADS = 8
NUM_LAYERS = 3
OUTPUT_DIM = len(class_values)

NUM_EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 3e-1
MOMENTUM = 0.9

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_acc", min_delta=1e-5, patience=5, restore_best_weights=True
)

# Build model
gat_model = anna.GraphAttentionNetwork(
    node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM, LEARNING_RATE, MOMENTUM
)

# Compile model
gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

gat_model.fit(
    x=train_indices,
    y=train_labels,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping],
    verbose=2,
)

_, test_accuracy = gat_model.evaluate(x=test_indices, y=test_labels, verbose=0)











































































