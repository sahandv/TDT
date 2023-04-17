#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:10:03 2020

@author: github.com/sahandv
"""
import sys
import time
import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randint

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics.cluster import silhouette_score,homogeneity_score,adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_mutual_info_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer , TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

from sciosci.assets import text_assets as ta
from DEC.DEC_keras import DEC_simple_run

#%% Extrinsic evaluation
# =============================================================================
def evaluate(X,Y,predicted_labels):
    
    df = pd.DataFrame(predicted_labels,columns=['label'])
    if len(df.groupby('label').groups)<2:
        return [0,0,0,0,0]
    try:
        sil = silhouette_score(X, predicted_labels, metric='euclidean')
    except:
        sil = 0
        
    return [sil,
            homogeneity_score(Y, predicted_labels),
            normalized_mutual_info_score(Y, predicted_labels),
            adjusted_mutual_info_score(Y, predicted_labels),
            adjusted_rand_score(Y, predicted_labels)]
# =============================================================================
# Cluster and evaluate
# =============================================================================
def run_extrinsic_tests(data_address:str,output_file_name:str,labels:list,k:int,algos:list):
    # clusters = labels.groupby('label').groups
    # k = len(list(clusters.keys()))
    # data_address = data_dir+file_name
    print('K=',k)
    print('Will write the results to',output_file_name)
    
    
    column_names = ['Method','parameter','Silhouette','Homogeneity','NMI','AMI','ARI']
    vectors = pd.read_csv(data_address)#,header=None,)
    try:
        vectors = vectors.drop('Unnamed: 0',axis=1)
        print('\nDroped index column. Now '+data_address+' has the shape of: ',vectors.shape)
    except:
        print('\nVector shapes seem to be good:',vectors.shape)
        
    # data_dir+file_name+' dm_concat'
    labels_f = pd.factorize(labels.label)

    X = vectors.values
    Y = labels_f[0]
    n_clusters = k
    
    assert X.shape[0]==Y.shape[0], "X Y dimension mismatch!"
    
    labels_task_1 = labels[(labels['label']=='car') | (labels['label']=='memory')]
    vectors_task_1 = vectors.iloc[labels_task_1.index]
    labels_task_1_f = pd.factorize(labels_task_1.label)
    X_task_1 = vectors_task_1.values
    Y_task_1 = labels_task_1_f[0]
    n_clusters_task_1 = 2
    
    results = pd.DataFrame([],columns=column_names)
    results_template = results.copy()
    
    
    # # =============================================================================
    # # Deep with min_max_scaling
    # # =============================================================================
    # archs = [[500, 500, 2000, 10],[500, 1000, 2000, 10],[500, 1000, 1000, 10],
    #             [500, 500, 2000, 100],[500, 1000, 2000, 100],[500, 1000, 1000, 100],
    #             [100, 300, 600, 10],[300, 500, 2000, 10],[700, 1000, 2000, 10],
    #             [200, 500, 10],[500, 1000, 10],[1000, 2000, 10],
    #             [200, 500, 100],[500, 1000, 100],[1000, 2000, 100],
    #             [1000, 500, 10],[500, 200, 10],[200, 100, 10],
    #             [1000, 1000, 2000, 10],[1000, 1500, 2000, 10],[1000, 1500, 1000, 10],
    #             [1000, 1000, 2000,500, 10],[1000, 1500, 2000,500, 10],[1000, 1500, 1000, 500, 10],
    #             [500, 500, 2000, 500, 10],[500, 1000, 2000, 500, 10],[500, 1000, 1000, 500, 10],
    #             [200,200,10],[200,200,10],[200,200,10],
    #             [200,200,10],[200,200,10],[200,200,10],
    #             [200,200,10],[200,200,10],[200,200,10],
    #             [200,500,10],[200,500,10],[200,500,10],
    #             [200,500,10],[200,500,10],[200,500,10]]
    # print('\n- DEC 2-----------------------')
    # for fold in tqdm(archs):
    #     predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=True,n_clusters=5,architecture=fold,pretrain_epochs=300)
    #     tmp_results = ['DEC minmax scaler',str(fold)]+evaluate(X,Y,predicted_labels)
    #     tmp_results = pd.Series(tmp_results, index = results.columns)
    #     results = results.append(tmp_results, ignore_index=True)
    # mean = results.mean(axis=0)
    # maxx = results.max(axis=0)
    # print(mean)
    # print(maxx)
    

    # =============================================================================
    # K-means
    # =============================================================================
    if 'kmeans-random' in algos:
        print('\n- k-means random -----------------------')
        for fold in tqdm(range(3)):
            seed = randint(0,10**5)
            model = KMeans(n_clusters=n_clusters,n_init=20, init='random', random_state=seed).fit(X)
            predicted_labels = model.labels_
            
            try:
                tmp_results = ['k-means random','seed '+str(seed)]+evaluate(X,Y,predicted_labels)
                tmp_results_s = pd.Series(tmp_results, index = results.columns)
                tmp_results_df = results_template.copy()
                tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
                results = results.append(tmp_results_s, ignore_index=True)
            except:
                print('Some error happened, skipping ',fold)
            
            print('writing the fold results to file')
            # if file does not exist write header 
            try:
                if not os.path.isfile(output_file_name):
                    tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
                else: # else it exists so append without writing the header
                    tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
                print('\nWirite success.')
            except:
                print('something went wrong and could not write the results to file!\n',
                     'You may abort and see what can be done.\n',
                     'Or wait to see the the final results in memory.')
                
            predicted_labels = pd.DataFrame(predicted_labels,columns=['labels'])
            pred_address = '/'.join(data_address.split('/')[:-1])+'/predictions/'
            predicted_labels.to_csv(pred_address+data_address.split('/')[-1]+' fold{fold}'.format(fold=fold),index=False)
        mean = results.mean(axis=0)
        maxx = results.max(axis=0)
        print(mean)
        print(maxx)
    # =============================================================================
    # K-means with init='k-means++'
    # =============================================================================
    if 'kmeans++' in algos:
        print('\n- k-means++ -----------------------')
        for fold in tqdm(range(5)):
            gc.collect()
            seed = randint(0,10**5)
            model = KMeans(n_clusters=n_clusters,n_init=20,init='k-means++', random_state=seed).fit(X)
            predicted_labels = model.labels_
            try:
                tmp_results = ['k-means++','seed '+str(seed)]+evaluate(X,Y,predicted_labels)
                tmp_results_s = pd.Series(tmp_results, index = results.columns)
                tmp_results_df = results_template.copy()
                tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
                results = results.append(tmp_results_s, ignore_index=True)
            except:
                print('Some error happened, skipping ',fold)
            
            print('writing the fold results to file')
            # if file does not exist write header 
            try:
                if not os.path.isfile(output_file_name):
                    tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
                else: # else it exists so append without writing the header
                    tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
                print('\nWirite success.')
            except:
                print('something went wrong and could not write the results to file!\n',
                     'You may abort and see what can be done.\n',
                     'Or wait to see the the final results in memory.')
        mean = results.mean(axis=0)
        maxx = results.max(axis=0)
        print(mean)
        print(maxx)
    # =============================================================================
    # Deep no min_max_scaling
    # =============================================================================
    if 'DEC' in algos:
        archs = [
                [500, 500, 2000, 500],[500, 500, 2000, 500],[500, 500, 2000, 500],
                [500, 500, 1000, 500],[500, 500, 1000, 500],[500, 500, 1000, 500],
                # [500, 500, 2000, 500],[500, 500, 2000, 500],[500, 500, 2000, 500],
                # [500, 1000, 2000, 100],[500, 1000, 2000, 100],[500, 1000, 2000, 100],
                [200, 1000, 2000,100, 10],[200, 1000, 2000,200, 10],[200, 1000, 2000, 500, 10],
                # [200, 500, 1000, 500, 10],[200, 500, 1000, 200, 10],[200, 500, 1000, 100, 10],
                # [200, 1000, 2000,100, 10],[200, 1000, 2000,200, 10],[200, 1000, 2000, 500, 10],
                # [200, 500, 1000, 500, 10],[200, 500, 1000, 200, 10],[200, 500, 1000, 100, 10],
                # [200, 1000, 2000,100],[200, 1000, 2000,200],[200, 1000, 2000, 500],
                # [200, 500, 1000, 500],[200, 500, 1000, 200],[200, 500, 1000, 100],
                # [200, 1000, 2000, 10],[200, 1000, 2000, 10],[200, 1000, 2000, 10],
                [1500,3000,1500,100],[1500,3000,1500,100],[1500,3000,1500,100],
                [300,1200,600,10],[300,1200,600,10],[300,1200,600,10],
                [300,1200,600,100,10],[300,1200,600,100,10],[300,1200,600,100,10],
                # [1024,1024,2048,256,10],[1024,1024,2048,256,10],
                # [512,1024,2048,128,10],[512,1024,2048,128,10],
                # [512,1024,2048,10],[512,1024,2048,10],
                # [1024,1024,2048,10],[1024,1024,2048,10],
                # [1536,768,384,192,10],[1536,768,384,192,10],[1536,768,192,10],
                [1500,3000,1500,100,10],[1500,3000,1500,100,10],[1500,3000,1500,100,10],
                # [1536,3072,1536,100],[1536,3072,1536,100],[1536,3072,1536,100],
                # [1536,3072,1536,10],[1536,3072,1536,10],[1536,3072,1536,10],
                # [1536,3072,1536,100,10],[1536,3072,1536,100,10],[1536,3072,1536,100,10],
                # [200,200,100],[200,200,100],[200,200,100],
                [200,500,20],[200,500,200],[200,500,200],
                # [200,200,10],[200,200,10],[200,200,10],
                # [400,400,10],[400,400,10],[400,400,10],
                # [400,400,10],[400,400,10],[400,400,10],
                [400,1000,10],[400,1000,10],[400,1000,100,10],[400,1000,100,10],
                [600,300,50,10],[600,300,50,10],[600,300,10],
                [400,500,10],[400,500,10],[400,500,10],
                # [200,200,10],[200,200,10],[200,200,10],
                # [200,200,10],[200,200,10],[200,200,10],
                [200,200,10],[200,200,10],[200,200,10]]
                # [200,500,10],[200,500,10],[200,500,10],
                # [200,500,10],[200,500,10],[200,500,10],
                # [200,500,10],[200,500,10],[200,500,10],
        archs = [
                [200,500,10],
                # [300,500,10],
                [500,500,10]]
        print('\n- DEC -----------------------')
        for fold in tqdm(archs):
            gc.collect()
            seed = randint(0,10**4)
            np.random.seed(seed)
            try:
                predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=False,n_clusters=n_clusters,architecture=fold,pretrain_epochs=200)
                tmp_results = ['DEC',str(seed)+' '+str(fold)]+evaluate(X,Y,predicted_labels)
                tmp_results_s = pd.Series(tmp_results, index = results.columns)
                tmp_results_df = results_template.copy()
                tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
                results = results.append(tmp_results_s, ignore_index=True)
            except:
                print('Some error happened, skipping ',fold)
            
            print('writing the fold results to file')
            # if file does not exist write header 
            try:
                if not os.path.isfile(output_file_name):
                    tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
                else: # else it exists so append without writing the header
                    tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
                print('\nWirite success.')
            except:
                print('something went wrong and could not write the results to file!\n',
                     'You may abort and see what can be done.\n',
                     'Or wait to see the the final results in memory.')
                
            predicted_labels = pd.DataFrame(predicted_labels,columns=['labels'])
            pred_address = '/'.join(data_address.split('/')[:-1])+'/predictions/'
            predicted_labels.to_csv(pred_address+data_address.split('/')[-1]+' fold{fold}'.format(fold=fold),index=False)
            
        mean = results.mean(axis=0)
        maxx = results.max(axis=0)
        print(mean)
        print(maxx)
        
    # =============================================================================
    # Agglomerative
    # =============================================================================
    if 'agglomerative' in algos:
        print('\n- Agglomerative -----------------------')
        for fold in tqdm(range(1)):
            gc.collect()
            model = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward').fit(X)
            predicted_labels = model.labels_
            try:
                tmp_results = ['Agglomerative','ward']+evaluate(X,Y,predicted_labels)
                tmp_results_s = pd.Series(tmp_results, index = results.columns)
                tmp_results_df = results_template.copy()
                tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
                results = results.append(tmp_results_s, ignore_index=True)
            except:
                print('Some error happened, skipping ',fold)
            
            print('writing the fold results to file')
            # if file does not exist write header 
            try:
                if not os.path.isfile(output_file_name):
                    tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
                else: # else it exists so append without writing the header
                    tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
                print('\nWirite success.')
            except:
                print('something went wrong and could not write the results to file!\n',
                     'You may abort and see what can be done.\n',
                     'Or wait to see the the final results in memory.')
        mean = results.mean(axis=0)
        maxx = results.max(axis=0)
        print(mean)
        print(maxx)
    # =============================================================================
    # DBSCAN
    # =============================================================================
    if 'DBSCAN' in algos:
        eps=0.000001
        print('\n- DBSCAN -----------------------')
        for fold in tqdm(range(5)):
            gc.collect()
            eps = eps+0.05
            model = DBSCAN(eps=eps, min_samples=10,n_jobs=15).fit(X)
            predicted_labels = model.labels_
            try:
                tmp_results = ['DBSCAN','eps '+str(eps)]+evaluate(X,Y,predicted_labels)
                tmp_results_s = pd.Series(tmp_results, index = results.columns)
                tmp_results_df = results_template.copy()
                tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
                results = results.append(tmp_results_s, ignore_index=True)
            except:
                print('Some error happened, skipping ',fold)
            
            print('writing the fold results to file')
            # if file does not exist write header 
            try:
                if not os.path.isfile(output_file_name):
                    tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
                else: # else it exists so append without writing the header
                    tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
                print('\nWirite success.')
            except:
                print('something went wrong and could not write the results to file!\n',
                      'You may abort and see what can be done.\n',
                      'Or wait to see the the final results in memory.')
        mean = results.mean(axis=0)
        maxx = results.max(axis=0)
        print(mean)
        print(maxx)

    # =============================================================================
    # Save to disk
    # =============================================================================
    # print('Writing to disk...')
    results_df = pd.DataFrame(results)
    # results_df.to_csv(output_file_name,index=False)
    print('Done.')
    return results_df


    
# =============================================================================
# Run
# =============================================================================
datapath = '/home/sahand/GoogleDrive/Data/'
# datapath = '/home/sahand/Projects/scioscipred/models/classification/vecs/best/'
data_dir =  datapath+"Corpus/cora-classify/cora/"
# data_dir =  datapath+""
# data_dir = datapath+'Corpus/Scopus new/embeddings/merged/d2v n2v 200D-10D exp11 z'

# label_address =  datapath+"Corpus/cora-classify/cora/clean/with citations new/corpus classes1"
label_address =  data_dir+"embeddings/single_component_small_19k/labels"
# label_address =  data_dir+"clean/with citations/corpus classes1"

# vec_file_names = ['embeddings/node2vec super-d2v-node 128-80-10 p4q1','embeddings/node2vec super-d2v-node 128-80-10 p1q025','embeddings/node2vec super-d2v-node 128-10-100 p1q025']#,'Doc2Vec patent corpus',
                  # ,'embeddings/node2vec-80-10-128 p1q0.5','embeddings/node2vec deepwalk 80-10-128']
# vec_file_names =  ['embeddings/node2vec super-d2v-node 128-80-10 p1q05']
vec_file_names =  [#'embeddings/single_component_small_18k/doc2vec 300D dm=1 window=10 tagged',
                   # 'embeddings/single_component_small_18k/doc2vec 300D dm=1 window=10']
                    # 'embeddings/with citations/merged VAE 200-100D exp01 z_mean',
                    # 'embeddings/with citations/merged VAE 200-100D exp01 z_log_var',
                    # 'embeddings/with citations/merged VAE 200-100D exp01 z',
                    # 'embeddings/with citations/node2vec 100D p2 q1 len5-50 04.10.2022 masked',                  
                    # 'embeddings/with citations/merged AE 200-10D exp01 z',
                   # 'embeddings/with citations/merged VAE 200-10D exp11 z',
                   # 'embeddings/with citations/merged VAE 200-10D exp11 z_log_var',
                   # 'embeddings/with citations/merged VAE 200-10D exp11 z_mean',
                   # 'embeddings/with citations/merged VAE 200-15D exp4 z',
                   # 'embeddings/with citations/merged VAE 200-15D exp4 z_log_var',
                   # 'embeddings/with citations/merged VAE 200-15D exp4 z_mean'
                    # 'doc2vec 300D dm=1 window=10-[200, 100, 10, False]-0.6673-0.6673-0.6631--vecs.csv',
                    # 'doc2vec 300D dm=1 window=10-[400, 200, 10, True]-0.6833-0.6833-0.6798--vecs.csv',
                    # 'DW 300-70-20-[200, 100, 10, False]-0.8073-0.8073-0.8079--vecs.csv',
                    # 'DW 300-70-20-[400, 200, 10, True]-0.8115-0.8115-0.8116--vecs.csv',
                    # 'DW 300-70-20 with_d2v300D_supernodes-[200, 100, 10, False]-0.8123-0.8123-0.8117--vecs.csv',
                    # 'DW 300-70-20 with_d2v300D_supernodes-[400, 200, 10, True]-0.8289-0.8289-0.8286--vecs.csv',
                    # 'naive d2v + dw 600-[200, 100, 10, False]-0.7887-0.7887-0.7879--vecs.csv',
                    # 'naive d2v + dw 600-[400, 200, 10, True]-0.7903-0.7903-0.7891--vecs.csv',
                    # 'naive d2v + n2v 600-[200, 100, 10, False]-0.7769-0.7769-0.7747--vecs.csv',
                    # 'naive d2v + n2v 600-[400, 200, 10, True]-0.7974-0.7974-0.7963--vecs.csv',
                    # 'node2vec 300-70-20 p1q05-[200, 100, 10, False]-0.8092-0.8092-0.8096--vecs.csv',
                    # 'node2vec 300-70-20 p1q05-[400, 200, 10, True]-0.8176-0.8176-0.8165--vecs.csv',
                    # 'node2vec 300-70-20 p1q05 with_d2v300D_supernodes-[200, 100, 10, False]-0.8097-0.8097-0.8092--vecs.csv',
                    # 'node2vec 300-70-20 p1q05 with_d2v300D_supernodes-[400, 200, 10, True]-0.8236-0.8236-0.8231--vecs.csv',
                    # 'TADW-120-240-tfidf-[200, 100, 10, False]-0.6945-0.6945-0.6914--vecs.csv',
                    # 'TADW-120-240-tfidf-[400, 200, 10, True]-0.7085-0.7085-0.7041--vecs.csv',
                    # 'TENE-150-300-tfidf-[200, 100, 10, False]-0.4003-0.4003-0.2293--vecs.csv',
                    # 'TENE-150-300-tfidf-[400, 200, 10, True]-0.3966-0.3966-0.2254--vecs.csv'
                    # 'doc2vec 300D dm=1 window=10-[400, 200, 10, False]-0.6869-0.6869-0.6838--vecs.csv',
                    # 'DW 300-70-20-[400, 200, 10, False]-0.8108-0.8108-0.8121--vecs.csv',
                    # 'DW 300-70-20 with_d2v300D_supernodes-[400, 200, 10, False]-0.8144-0.8144-0.8162--vecs.csv',
                    # 'naive d2v + dw 600-[400, 200, 10, False]-0.7942-0.7942-0.7933--vecs.csv',
                    # 'naive d2v + n2v 600-[400, 200, 10, False]-0.7982-0.7982-0.7969--vecs.csv',
                    # 'node2vec 300-70-20 p1q05-[400, 200, 10, False]-0.8081-0.8081-0.8072--vecs.csv',
                    # 'node2vec 300-70-20 p1q05 with_d2v300D_supernodes-[400, 200, 10, False]-0.8173-0.8173-0.8171--vecs.csv',
                    # 'TADW-120-240-tfidf-[400, 200, 10, False]-0.7115-0.7115-0.7081--vecs.csv',
                    # 'TENE-150-300-tfidf-[400, 200, 10, False]-0.4003-0.4003-0.2307--vecs.csv', 
                    'doc2vec 300D dm=1 window=10-[200, 100, 10, False]-0.6673-0.6673-0.6631--vecs.csv',
                    'doc2vec 300D dm=1 window=10-[400, 200, 10, False]-0.6869-0.6869-0.6838--vecs.csv',
                    'doc2vec 300D dm=1 window=10-[400, 200, 10, True]-0.6833-0.6833-0.6798--vecs.csv',
                    'DW 300-70-20-[200, 100, 10, False]-0.8073-0.8073-0.8079--vecs.csv',
                    'DW 300-70-20-[400, 200, 10, False]-0.8108-0.8108-0.8121--vecs.csv',
                    'DW 300-70-20-[400, 200, 10, True]-0.8115-0.8115-0.8116--vecs.csv',
                    'DW 300-70-20 with_d2v300D_supernodes-[200, 100, 10, False]-0.8123-0.8123-0.8117--vecs.csv',
                    'DW 300-70-20 with_d2v300D_supernodes-[400, 200, 10, False]-0.8144-0.8144-0.8162--vecs.csv',
                    'DW 300-70-20 with_d2v300D_supernodes-[400, 200, 10, True]-0.8289-0.8289-0.8286--vecs.csv',
                    'naive d2v + dw 600-[200, 100, 10, False]-0.7887-0.7887-0.7879--vecs.csv',
                    'naive d2v + dw 600-[400, 200, 10, False]-0.7942-0.7942-0.7933--vecs.csv',
                    'naive d2v + dw 600-[400, 200, 10, True]-0.7903-0.7903-0.7891--vecs.csv',
                    'naive d2v + n2v 600-[200, 100, 10, False]-0.7769-0.7769-0.7747--vecs.csv',
                    'naive d2v + n2v 600-[400, 200, 10, False]-0.7982-0.7982-0.7969--vecs.csv',
                    'naive d2v + n2v 600-[400, 200, 10, True]-0.7974-0.7974-0.7963--vecs.csv',
                    'node2vec 300-70-20 p1q05-[200, 100, 10, False]-0.8092-0.8092-0.8096--vecs.csv',
                    'node2vec 300-70-20 p1q05-[400, 200, 10, False]-0.8081-0.8081-0.8072--vecs.csv',
                    'node2vec 300-70-20 p1q05-[400, 200, 10, True]-0.8176-0.8176-0.8165--vecs.csv',
                    'node2vec 300-70-20 p1q05 with_d2v300D_supernodes-[200, 100, 10, False]-0.8097-0.8097-0.8092--vecs.csv',
                    'node2vec 300-70-20 p1q05 with_d2v300D_supernodes-[400, 200, 10, False]-0.8173-0.8173-0.8171--vecs.csv',
                    'node2vec 300-70-20 p1q05 with_d2v300D_supernodes-[400, 200, 10, True]-0.8236-0.8236-0.8231--vecs.csv',
                    'TADW-120-240-tfidf-[200, 100, 10, False]-0.6945-0.6945-0.6914--vecs.csv',
                    'TADW-120-240-tfidf-[400, 200, 10, False]-0.7115-0.7115-0.7081--vecs.csv',
                    'TADW-120-240-tfidf-[400, 200, 10, True]-0.7085-0.7085-0.7041--vecs.csv',
                    'TENE-150-300-tfidf-[200, 100, 10, False]-0.4003-0.4003-0.2293--vecs.csv',
                    'TENE-150-300-tfidf-[400, 200, 10, False]-0.4003-0.4003-0.2307--vecs.csv',
                    'TENE-150-300-tfidf-[400, 200, 10, True]-0.3963-0.3963-0.2252--vecs.csv'
                   ]
# vec_file_names = ['embeddings/with citations/n2v 100D p2 q1 len5-50']

# vec_file_names = [
#                 'embeddings/single_component_small/deep_nonlinear_embedding_600',
#                 'embeddings/single_component_small/doc2vec 300D dm=1 window=10',
#                 'embeddings/single_component_small/DW 300-70-20',
#                 'embeddings/single_component_small/DW 300-70-20 with_d2v300D_supernodes',
#                 'embeddings/single_component_small/node2vec 300-70-20 p1q05',
#                 'embeddings/single_component_small/node2vec 300-70-20 p1q05 with_d2v300D_supernodes',
#                 'embeddings/single_component_small/TADW-120-240',
#                 'embeddings/single_component_small/TENE-150-300-bow',
#                 'embeddings/single_component_small/naive d2v + dw 600',
#                 'embeddings/single_component_small/naive d2v + n2v 600'
#                 ]
vec_file_names = [
                   # 'embeddings/single_component_small_19k/doc2vec 300D dm=1 window=10',
                   
                   # 'embeddings/single_component_small_19k/node2vec 300-70-20 p1q05',
                   'embeddings/single_component_small_19k/node2vec 300-70-20 p1q05 with_d2v300D_supernodes',
                   
                   # 'embeddings/single_component_small_19k/DW 300-70-20',
                   'embeddings/single_component_small_19k/DW 300-70-20 with_d2v300D_supernodes',
                   
                   # 'embeddings/single_component_small_19k/naive d2v + dw 600',
                   # 'embeddings/single_component_small_19k/naive d2v + n2v 600',
                   
                   # 'embeddings/single_component_small_19k/TADW-120-240-tfidf',
                   # 'embeddings/single_component_small_19k/TENE-150-300-tfidf'
                   ]
# labels = pd.read_csv(label_address,names=['label'])
labels = pd.read_csv(label_address)
labels.columns = ['label']

clusters = labels.groupby('label').groups
algos = ['DEC']
for file_name in vec_file_names:
    gc.collect()
    output_file_name = data_dir+file_name+' clustering results'
    run_extrinsic_tests(data_dir+file_name,output_file_name,labels,len(list(clusters.keys())),algos)


# =============================================================================
#%% Performance evaluation
# =============================================================================

import pandas as pd
# datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/' #C1314
datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/' #Zen
# datapath = '/home/sahand/GoogleDrive/Data/' #Asus
data_address =  datapath+"Corpus/cora-classify/cora/embeddings/single_component_small/naive d2v + n2v 600 clustering results"
df = pd.read_csv(data_address)
# max1 = df.groupby(['Method'], sort=False).max()
# max2 = df.groupby(['Method']).agg({'NMI': 'max','AMI':'max','ARI':'max'})
max3 = df[df.groupby(['Method'])['ARI'].transform(max) == df['ARI']]
# min3 = df[df.groupby(['Method'])['NMI'].transform(min) == df['NMI']]

#%% Intrinsic evaluation - not completed
import sys
import time
import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from random import randint
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
from sklearn.decomposition import PCA

from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score
import scipy.cluster.hierarchy as sch
from sciosci.assets import advanced_assets as aa
seed = 0
random.seed(seed)

# mask = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/Scopus new/embeddings/concepts/empty_data')['idx'].values.tolist()
# data = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/Scopus new/embeddings/concepts/node2vec/100D p1 q05 len20 average of concepts',index_col=0)
data = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/Scopus new/embeddings/concepts/doc2vec/underlined 100D dm=1 window=12 gensim41')
data = pd.read_csv('/home/sahand/GoogleDrive/Data/Corpus/cora-classify/cora/embeddings/with citations/merged AE 200-10D exp01 z')
# index = pd.read_csv('~/Downloads/sample_idx',index_col=0).values[:,0].tolist()
# # index = pd.DataFrame(data.drop(mask,axis=0).sample(20000).index).to_csv('~/Downloads/sample_idx')
# data = data.loc[index].values
X = data.values
kmeans_results = []
dec_results = []



cluster_range =list(range(3,40,2))

print("Gridsearching the cluster ranges . . . ")
for n_clusters in tqdm(cluster_range,total=len(cluster_range)):
    silhouette = []
    sse = []
    dbi = []
    # clustering = AgglomerativeClustering(n_clusters=n_clusters,affinity='cosine',linkage='complete').fit(articles_vectors_filtered_np)
# =============================================================================
#     kmeans
# =============================================================================
    clustering = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    # clustering = AffinityPropagation().fit(article_vectors_np)
    cluster_labels = clustering.labels_
    kmeans_results.append([silhouette_score(X, cluster_labels),davies_bouldin_score(X, cluster_labels),calinski_harabasz_score(X, labels),sse])
# =============================================================================
# DEC
# =============================================================================
    # seed = randint(0,10**4)
    # np.random.seed(seed)
    arch = [200,500,10]
    predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=False,n_clusters=n_clusters,architecture=arch,pretrain_epochs=200)
    dec_results.append([silhouette_score(X, cluster_labels),davies_bouldin_score(X, cluster_labels)])
    # dbi.append(davies_bouldin_score(X, cluster_labels))
    # sse.append(clustering.inertia_)
    # silhouette_avg = silhouette_score(X, cluster_labels)
    # silhouette.append(silhouette_avg)

# fig = plt.figure()
# plt.plot(silhouette_avg_all)
# plt.title('doc2vec')
# plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Horizontally stacked subplots')
ax1.plot(x, y)
ax2.plot(x, -y)



plt.figure()
plt.plot(cluster_range, silhouette_avg_all)
plt.xlabel("Number of cluster")
# plt.ylabel("SSE")
plt.title('doc2vec')
plt.show()


dendrogram = aa.fancy_dendrogram(sch.linkage(data, method='ward'), truncate_mode='lastp',p=500,show_contracted=True,figsize=(15,8)) #single #average #ward


#%% Intrinsic evaluation

def DEC_minirun(X,results,results_template,n_clusters,fold,column_names,seed):
    predicted_labels = DEC_simple_run(X,minmax_scale_custom_data=False,n_clusters=n_clusters,architecture=fold,pretrain_epochs=200)
    dbi = davies_bouldin_score(X, predicted_labels)
    sil = silhouette_score(X, predicted_labels)
    chs = calinski_harabasz_score(X,predicted_labels)
    tmp_results = ['DEC',str(seed)+' '+str(fold),n_clusters,sil,dbi,chs,'']
    tmp_results_s = pd.Series(tmp_results, index = results.columns)
    tmp_results_df = results_template.copy()
    tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
    results = results.append(tmp_results_s, ignore_index=True)
    
    print('writing the fold results to file')
    # if file does not exist write header 
    try:
        if not os.path.isfile(output_file_name):
            tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
        else: # else it exists so append without writing the header
            tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
        print('\nWirite success.')
    except:
        print('something went wrong and could not write the results to file!\n',
             'You may abort and see what can be done.\n',
             'Or wait to see the the final results in memory.')
    
    return results

def run_intrinsic_tests(data_address:str,output_file_name:str,algos:list,k_b,k_e,k_s):
    print('Will write the results to',output_file_name)
    
    
    column_names = ['Method','parameter','n_clusters','Silhouette','DBI','CHS','SSE']
    vectors = pd.read_csv(data_address)#,header=None,)
    try:
        vectors = vectors.drop('Unnamed: 0',axis=1)
        print('\nDroped index column. Now '+data_address+' has the shape of: ',vectors.shape)
    except:
        print('\nVector shapes seem to be good:',vectors.shape)
    
    X = vectors.values
    results = pd.DataFrame([],columns=column_names)
    results_template = results.copy()
    
    # =============================================================================
    # DEC
    # =============================================================================
    
    if 'DEC' in algos:
        archs = [[200,500,10],[200,500,10]]
        print('\n- DEC -----------------------')
        for n_clusters in range(k_b,k_e,k_s):
            print(n_clusters)
            for fold in tqdm(archs):
                success = False
                tried = 0
                while success==False and tried<5:
                    gc.collect()
                    seed = randint(0,10**4)
                    np.random.seed(seed)
                    try:
                        results = DEC_minirun(X,results,results_template,n_clusters,fold,column_names,seed)
                        success=True
                    except Exception as e:
                        print('Some error happened, trying again ',n_clusters,fold,'--try',tried)
                        print(str(e))
                        # input('press enter to continue...')
                        tried+=1
                    
            mean = results.mean(axis=0)
            maxx = results.max(axis=0)
    # =============================================================================
    # K means
    # =============================================================================
    if 'kmeans-random' in algos:
        print('\n- k-means random -----------------------')
        for n_clusters in tqdm(range(k_b,k_e,k_s)):
            seed = randint(0,10**5)
            model = KMeans(n_clusters=n_clusters,n_init=20, init='random', random_state=seed).fit(X)
            predicted_labels = model.labels_
            try:
                dbi = davies_bouldin_score(X, predicted_labels)
                sil = silhouette_score(X, predicted_labels)
                chs = calinski_harabasz_score(X,predicted_labels)
                tmp_results = ['k-means random','seed '+str(seed),n_clusters,sil,dbi,chs,model.inertia_]
                tmp_results_s = pd.Series(tmp_results, index = results.columns)
                tmp_results_df = results_template.copy()
                tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
                results = results.append(tmp_results_s, ignore_index=True)
                
                print('writing the fold results to file')
                # if file does not exist write header 
                try:
                    if not os.path.isfile(output_file_name):
                        tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
                    else: # else it exists so append without writing the header
                        tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
                    print('\nWirite success.')
                except:
                    print('something went wrong and could not write the results to file!\n',
                         'You may abort and see what can be done.\n',
                         'Or wait to see the the final results in memory.')
            
            except:
                print('Some error happened, skipping ',n_clusters)
            
            mean = results.mean(axis=0)
            maxx = results.max(axis=0)
            print(mean)
            print(maxx)
    # =============================================================================
    # Agglomerative
    # =============================================================================
    if 'agglomerative' in algos:
        print('\n- Agglomerative -----------------------')
        for n_clusters in tqdm(range(k_b,k_e,k_s)):
            gc.collect()
            model = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward').fit(X)
            predicted_labels = model.labels_
            try:
                dbi = davies_bouldin_score(X, predicted_labels)
                sil = silhouette_score(X, predicted_labels)
                chs = calinski_harabasz_score(X,predicted_labels)
                tmp_results = ['Agglomerative','ward',n_clusters,sil,dbi,chs,'']
                tmp_results_s = pd.Series(tmp_results, index = results.columns)
                tmp_results_df = results_template.copy()
                tmp_results_df = tmp_results_df.append(tmp_results_s, ignore_index=True)
                results = results.append(tmp_results_s, ignore_index=True)
                
                print('writing the fold results to file')
                # if file does not exist write header 
                try:
                    if not os.path.isfile(output_file_name):
                        tmp_results_df.to_csv(output_file_name, header=column_names,index=False)
                    else: # else it exists so append without writing the header
                        tmp_results_df.to_csv(output_file_name, mode='a', header=False,index=False)
                    print('\nWirite success.')
                except:
                    print('something went wrong and could not write the results to file!\n',
                         'You may abort and see what can be done.\n',
                         'Or wait to see the the final results in memory.')
                
            except:
                print('Some error happened, skipping ',n_clusters)
            
            mean = results.mean(axis=0)
            maxx = results.max(axis=0)
            print(mean)
            print(maxx)
    # =============================================================================
    # Save to disk
    # =============================================================================
    # print('Writing to disk...')
    results_df = pd.DataFrame(results)
    # results_df.to_csv(output_file_name,index=False)
    print('Done.')
    return results_df
    
datapath = '/home/sahand/GoogleDrive/Data/'
# data_dir =  datapath+"Corpus/cora-classify/cora/"
data_dir =  datapath+"Corpus/Scopus new/"
# label_address =  datapath+"Corpus/cora-classify/cora/clean/with citations new/corpus classes1"

vec_file_names =  [ # 'embeddings/with citations/doc2vec 100D dm=1 mc3 window=12 gensim41',
                    # 'embeddings/TADW-120-240-tfidf',
                    # 'embeddings/TENE-120-240-tfidf',
                    'embeddings/n2v-300-240',
                    # 'embeddings/with citations/merged VAE 200-10D exp11 z',
                    # 'embeddings/with citations/merged VAE 200-10D exp11 z_log_var',
                    # 'embeddings/with citations/merged VAE 200-10D exp11 z_mean',
                    # 'embeddings/with citations/merged VAE 200-15D exp4 z',
                    # 'embeddings/with citations/merged VAE 200-15D exp4 z_log_var',
                    # 'embeddings/with citations/merged VAE 200-15D exp4 z_mean',
                   # '../../Scopus new/embeddings/merged/VAE d2v n2v 200D-10D exp11 z_mean',
                   # '../../Scopus new/embeddings/merged/VAE d2v n2v 200D-10D exp11 z_log_var',
                   # '../../Scopus new/embeddings/merged/VAE d2v n2v 200D-10D exp11 z',
                   # '../../Scopus new/embeddings/merged/AE 200-10D exp01 z',
                   # '../../Scopus new/embeddings/doc2vec 100D dm=1 window=12 gensim41',
                   # '../../Scopus new/embeddings/n2v 100D p2 q1 len5-50 with_id',
                    ]
# vec_file_names = ['embeddings/with citations/n2v 100D p2 q1 len5-50']

algos = ['kmeans-random']
r_df = []
r_s = []
for file_name in vec_file_names:
    gc.collect()
    output_file_name = data_dir+file_name+' clustering results intrinsic new'
    run_intrinsic_tests(data_dir+file_name,output_file_name,algos,k_b=8,k_e=12,k_s=2)
    
    
    
    
    
    
    
    
    
    
