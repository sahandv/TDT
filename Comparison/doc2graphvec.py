#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:20:41 2022

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

from sciosci.assets import graph_assets as gra
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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
n_type = 2 #network type selection
num_epochs = 40
vocab_limit = 40000
batch_size = 32
acc_all = list()
f1_all = list()
precision_all = list()
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
doc_vecs = pd.read_csv(datapath+'Corpus/Scopus new/embeddings/doc2vec 100D dm=1 window=12 gensim41 with_id')#'spaced 50D dm=1 mc3 window=12 gensim41 with_id')#Corpus/Scopus new/embeddings/doc2vec 300D dm=1 window=10 b3 gensim41
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
gc.collect()

# =============================================================================
# filter to the citations with data
# =============================================================================
if filter_refs:
    refs_filtered = []
    corpus_data_eids = corpus_data['file'].values.tolist()
    for i,row in tqdm(corpus_data.iterrows(),total=corpus_data.shape[0]):
        article_refs_filtered = []
        for ref in row['ref-eids']:
            if ref in corpus_data_eids:
                article_refs_filtered.append(ref)
        refs_filtered.append(article_refs_filtered)
        
    corpus_data['refs-filtered'] = refs_filtered
    mask = [True if len(x)>0 else False  for x in refs_filtered]
    corpus_data_b = corpus_data[mask] #backup
    corpus_data_masked = corpus_data_b.copy()
    corpus_data_masked['refs-filtered'] = corpus_data_masked['refs-filtered'].progress_apply(lambda x: '|'.join(x))
    corpus_data_masked['ref-ids'] = corpus_data_masked['ref-ids'].progress_apply(lambda x: '|'.join(x))
    corpus_data_masked['ref-eids'] = corpus_data_masked['ref-eids'].progress_apply(lambda x: '|'.join(x))
    corpus_data_masked.to_csv(datapath+'Corpus/Scopus new/clean/abstract_title method_b_3 with refs limited masked',index=False)
    corpus_data = corpus_data_b.copy()

# =============================================================================
# make the graph
# =============================================================================
G = nx.Graph()
for i,row in tqdm(corpus_data.iterrows(),total=corpus_data.shape[0]):
    for ref in row['refs-filtered']:
        G.add_edge(row['file'],ref)

# =============================================================================
#%% Prepare data - Cora
# =============================================================================
datapath = '/home/sahand/GoogleDrive/Data/'
data_dir =  datapath+'Corpus/cora-classify/cora/'
edges = pd.read_csv(data_dir+'citations.withabstracts')
label_address =  data_dir+'embeddings/with citations/corpus classes1'
labels = pd.read_csv(label_address,names=['class1'])
corpus_data = pd.read_csv(data_dir+'embeddings/with citations/doc2vec 100D dm=1 mc3 window=12 gensim41',index_col='id')
corpus_data = corpus_data.reset_index()
corpus_ids_backup = corpus_data.id.values.tolist()
corpus_data.drop('id',axis=1,inplace=True)

# =============================================================================
# Process labels
# =============================================================================
le = preprocessing.LabelEncoder()
le.fit(labels['class1'].values)
node_labels = le.transform(labels['class1'].values)


# =============================================================================
# translate cora id to sequential id
# =============================================================================
corpus_ids_dict = {id:i for i,id in enumerate(corpus_ids_backup)}
edges_filtered = []
for i,row in tqdm(edges.iterrows(),total=edges.shape[0]): #this is a mandatory filter. otherwise while changing into sequential ids, ids may conflict and overlap.
    if row.referring_id in corpus_ids_backup and row.cited_id in corpus_ids_backup:
        edges_filtered.append([row.referring_id,row.cited_id])
edges_filtered = pd.DataFrame(edges_filtered,columns=['referring_id','cited_id'])
edges_seq = edges_filtered.replace(corpus_ids_dict)

# =============================================================================
# make the graph
# =============================================================================
G = nx.Graph()
for i,row in tqdm(edges_seq.iterrows(),total=edges_seq.shape[0]):
    G.add_edge(row.referring_id,row.cited_id)
largest_cc = max(nx.connected_components(G), key=len)
largest_cc = G.subgraph(largest_cc)
len(list(G.edges()))
# =============================================================================
#%% Sample graph
# =============================================================================
# Number of walks
# Walk length
# P: Return hyperparameter
# Q: Inout hyperaprameter
n_walks = 50 
l_walks = 6 
p=0#2
q=1
s=1#5 # the factor of decreased chance of visit, if already visited
sf = 1#1.2 # the rate of increasing factor s, if visited multiple times
# n_workers = 15
worker_batch_size = 500

# Single threaded
walks = gra.sample(G,p,q,s,sf,l_walks)

# walks_id  = []
# for w in tqdm(walks):
#     walks_id.append([eid.split('-')[-1] for eid in w])
# =============================================================================
#%% Prepare nodal data sequences
# =============================================================================
full_data = []
walk_start_ids = []
walk_neighbour_ids = []
walk_end_ids = []
for walk in tqdm(walks):
    walk_start_ids.append(walk[0])
    walk_neighbour_ids.append(walk[1])
    walk_end_ids.append(walk[-1])
    full_data.append([corpus_data.loc[node] for node in walk[:-1]])
X1 = np.array(full_data)
X2 = np.array(walk_start_ids)
Y2 = np.array(walk_neighbour_ids)
Y = np.array(walk_end_ids)
Y_label = np.array([node_labels[i] for i  in Y])

# =============================================================================
#%% Split data
# =============================================================================
train_mask,validation_mask,test_mask = anna.mask_maker(data_range=X1.shape[0],train_split=0.8,test_split=0.05)

x_train = X1[train_mask]
x_validation = X1[validation_mask]
x_test = X1[test_mask]

x_train_id = X2[train_mask]
x_validation_id = X2[validation_mask]
x_test_id = X2[test_mask]

y_train = Y[train_mask]
y_validation = Y[validation_mask]
y_test = Y[test_mask]

y_train_neighbour = Y2[train_mask]
y_validation_neighbour = Y2[validation_mask]
y_test_neighbour = Y2[test_mask]

y_train_label = Y_label[train_mask]
y_validation_label = Y_label[validation_mask]
y_test_label = Y_label[test_mask]

n_classes = np.unique(Y_label).shape[0]

# del X1
gc.collect()

# train_data_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_train, y_train_label, length=x_train.shape[1],stride=1, batch_size=64)
# validation_data_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_validation, y_validation_label, length=x_validation.shape[1],stride=1, batch_size=64)
# test_data_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_test, y_test_label, length=x_test.shape[1],stride=1, batch_size=64)

# =============================================================================
#%% prediction task - citation prediction
# =============================================================================
n_features = corpus_data.shape[1]
n_input = x_train.shape[1]
n_docs = corpus_data.shape[0]
n_type=1
try:
    del model
except:
    print('no models found')
if n_type==1:
    # Only network input
    inputs_seq = tf.keras.Input(shape=(n_input,n_features,), name='input_1')
    # ADD embedding
    h = tf.keras.layers.LSTM(n_features,return_sequences=True,name='node_LSTM1')(inputs_seq)
    h = tf.keras.layers.LSTM(n_features,return_sequences=False,name='node_LSTM2')(h)
    h = tf.keras.layers.Dropout(0.2)(h)
    h = tf.keras.layers.Dense(n_features*n_input*2,name='dense1')(h)
    # ADD CRF
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax",name='dense2')(h)
    
    # Build model
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = tf.keras.Model(inputs=inputs_seq, outputs=outputs)
    accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/classification-lstm.h5', monitor='val_acc', mode='max', save_best_only=True)
    callback = keras.callbacks.EarlyStopping(
        monitor="val_acc", min_delta=1e-5, patience=5, restore_best_weights=True
    )
    tensorboard = tf.keras.callbacks.TensorBoard(
                              log_dir='.\logs',
                              histogram_freq=1,
                              write_images=True
                            )
    model.compile(loss=loss_fn,optimizer='adam',metrics=[accuracy_fn])
    model.summary()


if n_type==2:
    # Only network input
    n_classes = n_docs
    inputs_seq = tf.keras.Input(shape=(n_input,n_features,), name='input_1')
    # ADD embedding
    h = tf.keras.layers.LSTM(n_features,return_sequences=True,name='node_LSTM1')(inputs_seq)
    h = tf.keras.layers.LSTM(n_features,return_sequences=False,name='node_LSTM2')(h)
    h = tf.keras.layers.Dropout(0.2)(h)
    h = tf.keras.layers.Dense(n_features*n_input*2,name='dense1')(h)
    # ADD CRF
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax",name='dense2')(h)
    
    # Build model
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = tf.keras.Model(inputs=inputs_seq, outputs=outputs)
    accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/classification-lstm.h5', monitor='val_acc', mode='max', save_best_only=True)
    callback = keras.callbacks.EarlyStopping(
        monitor="val_acc", min_delta=1e-5, patience=5, restore_best_weights=True
    )
    
    tensorboard = tf.keras.callbacks.TensorBoard(
                              log_dir='.\logs',
                              histogram_freq=1,
                              write_images=True
                            )
    model.compile(loss=loss_fn,optimizer='adam',metrics=[accuracy_fn])
    model.summary()


if n_type==3:
    n_classes = n_docs
    inputs_doc = tf.keras.Input(shape=(1,), name='input_2')
    x_1 = tf.keras.layers.Embedding(n_docs,embedding_dim,input_length=1,name='doc_embedding')(inputs_doc)
    x_1 = tf.keras.layers.Flatten(name='doc_flatten')(x_1)
    x_1 = tf.keras.layers.Dense(embedding_dim,activation='relu')(x_1)
    x_1_model = tf.keras.Model(inputs=inputs_doc, outputs=x_1)
    
    
    inputs_netvec = tf.keras.Input(shape=(n_input,n_features,), name='input_1')
    x_2 = tf.keras.layers.LSTM(n_features,return_sequences=True,name='node_LSTM1')(inputs_netvec)
    x_2 = tf.keras.layers.LSTM(n_features,return_sequences=False,name='node_LSTM2')(x_2)
    x_2 = tf.keras.layers.Dropout(0.2)(x_2)
    x_2 = tf.keras.layers.Dense(n_features*n_input*10,name='dense1')(x_2)
    x_2_model = tf.keras.Model(inputs=inputs_netvec, outputs=x_2)
    
    x = tf.keras.layers.concatenate([x_1_model.output, x_2_model.output], name='concatenate')
    x = tf.keras.layers.Dense(1000,activation='relu',name='main_dense_1')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax",name='final_dense')(x)
    
    model = keras.Model(inputs=[x_1_model.input, x_2_model.input], outputs=outputs)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
    model.compile(loss=loss_fn,optimizer='adam',metrics=[accuracy_fn])
    tf.keras.utils.plot_model(model, to_file='combined_embedding.png', show_shapes=True, show_layer_names=True)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=35)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/classification.h5', monitor='val_accuracy', mode='max', save_best_only=True)
    model.summary()
    tensorboard = tf.keras.callbacks.TensorBoard(
                              log_dir='.\logs',
                              histogram_freq=1,
                              write_images=True
                            )

# =============================================================================
#%% Train single input
# =============================================================================

history = model.fit(x_train,y_train_label,
                    epochs=20, 
                    validation_data=(x_validation,y_validation_label),
                    verbose=1,
                    batch_size=64,
                    callbacks=[callback,checkpoint,tensorboard])

predictions = model.predict(x_test)#y_test_label
anna.plot_graphs(history,'acc',save='./models/classification-lstm.png')
predictions_label = np.argmax(predictions,axis=1)
model.save('./models/embedding-lstm.h5')

acc,f1,precision = anna.test_evaluate(predictions_label,y_test_neighbour)
print(np.around(acc*100,2),'&',np.around(f1*100,2),'&',np.around(precision*100,2))

acc_all.append(acc)
f1_all.append(f1)
precision_all.append(precision)

print(np.around(np.mean(acc_all)*100,2),'&',np.around(np.mean(f1_all)*100,2),'&',np.around(np.mean(precision_all)*100,2))

# =============================================================================
#%% Train multi input
# =============================================================================

history = model.fit([x_train_id,x_train],y_train_neighbour,
                    epochs=num_epochs, 
                    validation_data=([x_validation_id,x_validation],y_validation_neighbour),
                    verbose=1,
                    batch_size=batch_size,
                    callbacks=[callback,checkpoint,tensorboard])
model.save('./models/embedding-multi.h5')
predictions = model.predict([x_test_id,x_test])#y_test_label
anna.plot_graphs(history,'acc',save='./models/classification-lstm.png')
predictions_label = np.argmax(predictions,axis=1)

acc,f1,precision = anna.test_evaluate(predictions_label,y_test_neighbour)
print(np.around(acc*100,2),'&',np.around(f1*100,2),'&',np.around(precision*100,2))

acc_all.append(acc)
f1_all.append(f1)
precision_all.append(precision)

print(np.around(np.mean(acc_all)*100,2),'&',np.around(np.mean(f1_all)*100,2),'&',np.around(np.mean(precision_all)*100,2))
#%% Get embeddings (multi input)

weights = model.get_layer('dense1').get_weights()[0]
weights = pd.DataFrame(weights)
weights['id'] = corpus_ids_backup
weights.to_csv('/home/sahand/GoogleDrive/Data/Corpus/cora-classify/cora/embeddings/with citations/LSTM-merge',index=False)

#%% Prepare refs

refs_path = '/home/sahand/Documents/scopus_refs/REF'
datapath = '/home/sahand/GoogleDrive/Data/' #Ryzen
doc_vecs = pd.read_csv(datapath+'Corpus/Scopus new/embeddings/doc2vec 100D dm=1 window=12 gensim41')#Corpus/Scopus new/embeddings/doc2vec 300D dm=1 window=10 b3 gensim41
# corpus_data_ = pd.read_csv(datapath+'Corpus/Scopus new/clean/abstract_title method_b_3') # get the pre-processed abstracts
corpus_data = pd.read_csv(datapath+'Corpus/Scopus new/clean/abstract_title method_b_3 with_ref_file_address') # get the pre-processed abstracts

errors = []
refs = []
ref_ids = []
ref_eids = []
for i,row in tqdm(corpus_data.iterrows(),total=corpus_data.shape[0]):
    article_refs_ids = []
    article_refs_eids = []
    try:
        full_path = refs_path+'/'+row['file']
        with open(full_path) as f:
            jf = json.load(f)
            
        try:
            article_refs = jf[list(jf.keys())[0]]['references']['reference']
            if type(article_refs) is dict: # unify formats, so we always have dicst in  a list, even if 1.
                article_refs = [article_refs]
            refs.append(article_refs)
            for r in article_refs:
                try:
                    article_refs_ids.append(r['scopus-id'])
                except :
                    errors.append([i,'id not found'])
            for r in article_refs:
                try:
                    article_refs_eids.append(r['scopus-eid'])
                except :
                    errors.append([i,'eid not found'])
                    
        except:
            errors.append([i,'article refs format mismatch'])
            refs.append([])
            print('article refs format mismatch')
            
    except:
        errors.append([i,'no_data'])
        print('no data')
        refs.append([])
    
    ref_ids.append(article_refs_ids)
    ref_eids.append(article_refs_eids)

corpus_data['ref-ids'] = ['|'.join(x) for x in ref_ids]
corpus_data['ref-eids'] = ['|'.join(x) for x in ref_eids]
corpus_data.to_csv(datapath+'Corpus/Scopus new/clean/abstract_title method_b_3 with refs',index=False)


#%% Fix ref names and ids
dir_path  = ''
lens = []
ref_files = []
ref_files_ids = []
for path in os.listdir(dir_path):
    ref_files.append(path)
    ref_files_ids.append(int(path.split('-')[-1]))
    lens.append(len(path.split('-')[-1]))
    
id_file_map = pd.DataFrame({'file':ref_files,'id':ref_files_ids})

corpus_ids = corpus_data['id'].values.tolist()
id_file_map_filtered = id_file_map[id_file_map['id'].isin(corpus_ids)]
corpus_data_full = corpus_data.join(id_file_map_filtered.set_index('id'),on='id',how='left',lsuffix='_l')
corpus_data_full[pd.isnull(corpus_data_full['file'])]
corpus_data_full.to_csv(datapath+'Corpus/Scopus new/clean/abstract_title method_b_3 with_ref_file_address',index=False)

