#%% Algorithm
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 08:33:08 2021

@author: github.com/sahandv
"""
import os
import sys
import gc
import copy
import random
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
from scipy import spatial
import logging
import json
import itertools
from itertools import chain
from collections import defaultdict
import argparse
import pickle

from sklearn.metrics.cluster import silhouette_score,homogeneity_score,adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score,adjusted_mutual_info_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from gensim.models import FastText as fasttext_gensim
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

from sciosci.assets import text_assets as ta
from sciosci.assets import advanced_assets as aa

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# from DEC.DEC_keras import DEC_simple_run

pd.options.mode.chained_assignment = None  # default='warn'
logger = logging.getLogger(__name__)
tqdm.pandas()

# =============================================================================
# Evaluation method
# ===========================k==================================================
def evaluate(X,Y,predicted_labels):
    
    df = pd.DataFrame(predicted_labels,columns=['label'])
    if len(df.groupby('label').groups)<2:
        return [0,0,0,0,0,0]
    
    try:
        sil = silhouette_score(X, predicted_labels, metric='euclidean')
    except:
        sil = 0
        
    return [sil,
            homogeneity_score(Y, predicted_labels),
            homogeneity_score(predicted_labels, Y),
            normalized_mutual_info_score(Y, predicted_labels),
            adjusted_mutual_info_score(Y, predicted_labels),
            adjusted_rand_score(Y, predicted_labels)]

# =============================================================================
# Prepare
# =============================================================================
def plot_3D(X,labels,predictions,opacity=0.7):
    """
    Parameters
    ----------
    X : 2D np.array 
        Each row is expected to have values for X, Y, and Z dimensions.
    labels : iterable
        A 1D iterable (list or array) for labels of hover names.
    predictions : iterable
        A 1D iterable (list or array) for class labels. Must not be factorized.
    """
    X_df = pd.DataFrame(X)
    X_df['class'] = predictions
    X_df.columns = ['x_ax','y_ax','z_ax','class']
    X_df = X_df.reset_index()
    X_df['labels'] = labels
    # X_grouped = X_df.groupby('class').groups
    
    fig = px.scatter_3d(X_df, x='x_ax', y='y_ax',z='z_ax', color='class', opacity=opacity,hover_name='labels') #.iloc[X_grouped[i]]
    plot(fig)

def get_abstract_keywords(corpus,keywords_wanted,max_df=0.9,max_features=None):
    stop_words = set(stopwords.words("english"))
    cv=CountVectorizer(max_df=max_df,stop_words=stop_words, max_features=max_features, ngram_range=(1,1))
    X=cv.fit_transform(corpus)
    # get feature names
    feature_names=cv.get_feature_names()
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    keywords_tfidf = []
    keywords_sorted = []
    for doc in tqdm(corpus,total=len(corpus)):
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
        sorted_items=ta.sort_coo(tf_idf_vector.tocoo())
        keywords_sorted.append(sorted_items)
        keywords_tfidf.append(ta.extract_topn_from_vector(feature_names,sorted_items,keywords_wanted))
    return keywords_tfidf

def get_corpus_top_keywords(abstract_keywords_dict=None):
    if abstract_keywords_dict == None:
        print("keywords should be provided")
        return False
    terms = []
    values = []
    for doc in abstract_keywords_dict:
        if doc != None:
            terms = terms+list(doc.keys())
            values = values+list(doc.values())
    terms_df = pd.DataFrame({'terms':terms,'value':values}).groupby('terms').sum().sort_values('value',ascending=False)
    return terms_df

def find_max_item_value_in_all_cluster(haystack,needle,cluster_exception=None):
    max_val = 0
    max_index = None
    counter = 0
    for item in haystack:
        try:
            if item[needle]>max_val:
                if cluster_exception==None:
                    max_val = item[needle]
                    max_index = counter
                else:
                    if cluster_exception != counter:
                        max_val = item[needle] 
                        max_index = counter
        except:
            pass
        counter+=1

        if max_index!=None:
            row_max = haystack[max_index][list(haystack[max_index].keys())[0]] # Will give the maximum value (first item) of the row with max value of the needle. This gives us a perspective to see how this score compares to the max in the same row.
        else:
            row_max = 0
    # except:
        # row_max = None
    return max_val,row_max

def kw_to_string(kw:list):
    kw = [k.replace(' ','_') for k in kw]
    return ' '.join(kw)


class OGC:
    """
        Initialization Parameters
        ----------
        k : int, optional
            - The number of clusters for initial time slot. The default is 5.
        tol : float, optional
            - Tolerance for centroid stability measure. The default is 0.00001.
        n_iter : int, optional
            - Maximum number of iterations. The default is 300.
        patience : int, optional
            - patience for telerance of stability. It is served as the minimum 
            number of iterations after stability of centroids to continue the 
            iters. The default is 2.
        boundary_epsilon_coeff : float, optional, 
            - Coefficient of the boundary Epsilon, to calculate the exact epsilon 
            for each cluster. Used to assign a node to a cluster out of the 
            boundary of current nodes. 
            - Epsilon=Radius*boundary_epsilon_coeff
            - If zero, will not have evolutions.
            - The default is 0.1.
        boundary_epsilon_abs : float, optional
            **** DEPRECATED ****
            - Absolute value for boundary epsilon. 
            - If None, will ignore and use adaptive (boundary_epsilon_coeff) for calculations.
            - The default is None.
        minimum_nodes : int, optional
            - Minimum number of nodes to make a new cluster. The default is 10.
        a : float, optional, 
            - Weight or slope of temporal distance, while a>=0. 
            - The effectiveness of node in centroid calculation will be calculated 
            as in a weight function such as the default function ( V*[1/((a*t)+1)]), 
            where t is time delta, and V is vector value.
            - The default is 1.
        kernel_coeff : float, optional,
            - Density kernel is computed as the minimum radius value of the clusters. 
            The coefficient is multiplied with the computed value to yield the final 
            kernel bandwith as: bandwith=min(radius)*kernel_coeff
            - The default is 2.
        death_threshold : int, optional,
            - Classes with population below this amount will be elimined.
        growth_threshold_population: float, optional
            - Population in each class should grow over this amount to be considered a growing cluster, population-wise.
            - Default is 1.1
        max_keyword_distance : float, optional,
            - Distance threshold for validity of keyword ontology term distance.
            - The default is 0.50.
        growth_threshold_radius: float, optional
            - Radius in each class should grow over this amount to be considered a growing cluster, area-wise.
            - Default is 1.1
        merge_sub_clusters: bool, optional
            - If True, will consider the freshly sub-clustered clusters for merging as well.
            - Default is False
        merge_split_iters: int, optional
            - Iterations to perform the splitting and merging operations consecutively in a time-slice.
            - Default is 1
        merge_distance_percentage: float, optional
            - Coefficient of distance. New inter-cluster distance will be compared to the product of this value and the 
            previous distance to trigger merger.
            - The default is 1.0.
        split_regression_offset: float, optional
            - usually should be the same as the lowest split vote, to pull the votes up to over zero
        seed : int, optional
            - If seed is set, a seed will be used to make the results reproducable. 
            - The default is None.
        initializer : str, optional
            - Centroid initialization. The options are:
                'random_generated' randomly generated centroid based on upper and lower bounds of data.  
                'random_selected' randomly selects an existing node as initialization point.
                The default is 'random_generated'.
        distance_metric : str, optional
            - Options are 'euclidean' and 'cosine' distance metrics. The default is 'euclidean'.
        verbose : int, optional
            - '1' verbosity outputs main iterations and steps
            - '2' verbosity outputs debug data within each iteration, such as distances for tolerance
            - The default is 1.
            
        Class Variables
        ----------
        centroids_history: list
            List of centroids in each iteration. Cannot be used to plot the evolution.
        centroids_history: list
            List of centroids in each iteration. Cannot be used to plot the evolution.
        centoids: dict
            Current centroid values. Dict is used to preserve centroid id after classification eliminations/additions.
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t'). 
            
        Upcoming changes
        ----------
        Weighted nodes will be added, This can  help distinguish linked concepts BBBC from BC in a document.
        Impurity measure will be added: measure if new concepts have been added to a cluster, with what population.
        
            
        """
    def __init__(self, k:int=5, tol:float=0.00001, n_iter:int=300, patience=2, cumulative=False,
                 boundary_epsilon_coeff:float=0.1, boundary_epsilon_abs:float=None, 
                 boundary_epsilon_growth:float=0, minimum_nodes:int=10, seed=None, split_auto_threshold:float=2450, merge_auto_threshold:float=2100, #split_auto_threshold:float=2178, merge_auto_threshold:float=2225
                 a:float=1.0, kernel_coeff:float=2,death_threshold:int=10, levels_to_erode:int=50000, levels_to_erode_min:int=0,
                 growth_threshold_population:float=1.1,growth_threshold_radius:float=1.1,merge_distance_percentage:float=1.0,
                 max_keyword_distance:float=0.50,merge_sub_clusters:bool=False,merge_split_iters:int=1,
                 split_regression_coefficient:float=-0.7147367793518765,split_regression_offset:float=-48000,#7014774189438925
                 merge_regression_coefficient:float=0.7047367793518765,merge_regression_offset:float=2000, #7234744882046347
                 initializer:str='random_generated',distance_metric:str='euclidean',auto_split:bool=True,auto_merge:bool=True,
                 v:int=1,log:bool=False):
        
        self.k = k
        self.cumulative = cumulative
        self.tol = tol
        self.n_iter = n_iter
        self.centroids_history = []
        self.centroids_history_t = []
        self.centroids = {}
        self.seed = seed
        self.initializer = initializer
        self.distance_metric = distance_metric
        self.boundary_epsilon_coeff = boundary_epsilon_coeff
        self.minimum_nodes = minimum_nodes
        self.patience = patience
        self.a=a
        self.boundary_epsilon_growth = boundary_epsilon_growth
        self.kernel_coeff = kernel_coeff
        self.death_threshold = death_threshold
        self.growth_threshold_population = growth_threshold_population
        self.growth_threshold_radius = growth_threshold_radius
        self.max_keyword_distance = max_keyword_distance
        self.ontology_search_index = {}
        self.split_auto_threshold = split_auto_threshold
        self.merge_auto_threshold = merge_auto_threshold
        self.levels_to_erode = levels_to_erode
        self.levels_to_erode_min = levels_to_erode_min
        self.evolution_event_counter = 0
        self.evolution_events = {}
        self.temp_all = {}
        self.temp = defaultdict(lambda:[])
        self.classifications_log = {}
        self.merge_sub_clusters = merge_sub_clusters
        self.merge_split_iters = merge_split_iters
        self.split_regression_coefficient = split_regression_coefficient
        self.merge_regression_coefficient = merge_regression_coefficient
        self.split_regression_offset = split_regression_offset
        self.merge_regression_offset = merge_regression_offset
        self.merge_distance_percentage = merge_distance_percentage
        self.auto_split = auto_split
        self.auto_merge = auto_merge
        self.t = 0
        self.v = v
        self.log = log
        self.event_log = []
        self.populations = {}
        if seed != None:
            np.random.seed(seed)
    
    def verbose(self,value,**outputs):
        """
        Verbose outputs

        Parameters
        ----------
        value : int
            An integer showing the verbose level of an output. Higher verbose value means less important.
            If the object's initialized verbose level is equal to or lower than this value, it will print.
        **outputs : kword arguments, 'key'='value'
            Outputs with keys. Keys will be also printed. More than one key and output pair can be supplied.
        """
        if self.log:
            for output in outputs.items():
                if value<=self.v:
                    print('\n> '+str(output[0])+':',output[1])
                self.event_log.append([str(datetime.now())+'\n> '+str(output[0])+':',output[1]])
                
        
    def set_keyword_embedding_model(self,model):
        self.model_kw = model
    
    def set_ontology_tree(self,G):
        self.ontology_tree = G
    
    def set_ontology_keyword_search_index(self,index:dict):
        self.ontology_search_index = index
    
    def set_ontology_dict(self,ontology):
        self.ontology_dict_base = ontology
    
    def vectorize_keyword(self,keyword):
        return np.array([self.model_kw.wv[key] for key in keyword.split(' ')]).mean(axis=0)
    
    def prepare_ontology(self):
        """
        Format the ontologies so each key will direct to its vector and level 2 parents efficiently

        """
        self.ontology_dict = {}
        for key in tqdm(self.ontology_dict_base):
            self.ontology_dict[key]={'parents':self.ontology_dict_base[key],'vector':self.vectorize_keyword(key)}    
        return self.ontology_dict
    
    def map_keyword_ontology(self,keyword,max_keyword_distance:float=None):
        """
        Will return the most similar concept for a given keyword. Basically a search engine.
        """
        if max_keyword_distance==None:
            max_keyword_distance = self.max_keyword_distance
        
        distances = [self.get_distance(self.vectorize_keyword(keyword),self.ontology_dict[i]['vector'],self.distance_metric) for i in self.ontology_dict]
        item_index = np.argmin(np.array(distances))
        if np.min(np.array(distances))<=max_keyword_distance and len(keyword)>0:
            return list(self.ontology_dict.keys())[item_index]
        else:
            # print(False)
            return False

    def map_keyword_ontology_from_index(self,keyword,max_keyword_distance:float=None):
        """
        Will return the most similar concept for a given keyword from a pre-indexed search dictionary.
        """
        if max_keyword_distance==None:
            max_keyword_distance = self.max_keyword_distance
        try:
            if self.ontology_search_index[keyword][1]<=max_keyword_distance:
                return self.ontology_search_index[keyword][0]
            else:
                self.verbose(3,debug="Keyword does not meet threshold : "+str(keyword)+" - "+str(self.ontology_search_index[keyword][1])+'<'+str(max_keyword_distance))
                return False
        except:
            self.verbose(3,debug="Keyword does not exist in index: "+str(keyword))
            return False

    def return_root_doc_vec(self,root:str,classifications_portion,ignores:list):
        f_debug_code = random.randint(1, 1000)
        self.verbose(value=3,debug='root '+str(root))
        self.verbose(value=5,debug='classifications_portion '+str(classifications_portion))
        self.verbose(value=4,debug='ignores '+str(ignores))
        for i,row in classifications_portion.iterrows():
            if i not in ignores:
                if root in list(row['roots']):
                    self.temp['root_selection_return'] = {'data':row,'i':i,'t':self.t,'loop_id':f_debug_code}
                    return row[self.columns_vector].values,i
                    
                
        self.verbose(value=4,debug='did not find a proposal!!! ')
        self.temp['root_selection_return fail'] = {'data':{'root':root,'classifications_portion':classifications_portion},'t':self.t,'loop_id':f_debug_code}

    def multigraph_to_graph(self,M):
        G = nx.Graph()
        for u,v in M.edges():
            if G.has_edge(u,v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
        return G

    def graph_component_test(self,G_temp,ratio_thresh:float=1/10,count_trhesh_low:int=10) -> list:

        """
        Parameters
        ----------
        G_temp : networkx graph
        ratio_thresh : float, optional
            Ratio of populations to eachother to be considered significant. The default is 1/10.
        count_trhesh_low : int, optional
            Minimum number of population to be considered. The default is 10. 
            It is compared to the number of edges. Number of edges is basically the number of papers connecting the topics.

        Returns
        -------
        concept_proposals : list
            list of disjoint concepts to be considered for splitting.
        """
        self.temp['G'] = {'data':G_temp,'t':self.t}
        if nx.number_connected_components(G_temp)>1:
            self.verbose(4,debug=' -  -  - concept graph in class has '+str(nx.number_connected_components(G_temp))+' connected components.')
            sub_graphs = list(G_temp.subgraph(c) for c in nx.connected_components(G_temp))
            self.temp['sub_graphs'] = {'data':sub_graphs,'t':self.t}
            edges_counts = [c.number_of_edges() for c in sub_graphs]
            self.temp['edges_counts'] = {'data':edges_counts,'t':self.t}
            if all(v == 0 for v in edges_counts):
                self.verbose(4,debug=' -  -  - all(v == 0 for v in edges_counts) is True . edges_counts:'+str(edges_counts)+' .')
                return list(G_temp.nodes)
            
            edges_counts_l = [x for x in edges_counts if x>count_trhesh_low]
            self.verbose(2,debug=' -  -  - concept subgraph edge counts are: '+str(edges_counts_l))
            
            to_split = [x for x in list(itertools.combinations(edges_counts_l, 2)) if min(x[0],x[1])/max(x[0],x[1])>=ratio_thresh]
            self.verbose(4,debug=' -  -  - found splittable components: '+str(len(list(itertools.chain.from_iterable(to_split)))))
            to_split = list(itertools.chain.from_iterable(to_split))
            to_split_indices = [edges_counts.index(count) for count in to_split]
            concept_proposals = [list(sub_graphs[i].nodes())[0] for i in to_split_indices]
            return list(set(concept_proposals))
        else:
            self.verbose(4,debug=' -  -  - concept graph in class has '+str(nx.number_connected_components(G_temp))+' connected components. Skipping.')
            return []

    def visualize_graph(self,G,path:str="graph.png",color:str="red",shape:str="circle"):
        A = nx.nx_agraph.to_agraph(G)
        for i, node in enumerate(A.iternodes()):
                node.attr['shape'] = 'ellipse'
        A.node_attr["shape"] = shape
        A.edge_attr["color"] = color
        A.layout(prog="dot")
        A.draw(path)

    def erosion_component_test(self,G_tmp,nodes,level:int=20,ratio_thresh:float=1/10,count_trhesh_low:int=10):
        G_temp = copy.deepcopy(G_tmp)
        to_delete = [list(x) for x in list(itertools.combinations(nodes, 2))]
        self.verbose(4,debug=' -  -  - performing edge erosion level '+str(level))
        for i in range(level):
            G_temp.remove_edges_from(to_delete)
        self.verbose(4,debug=' -  -  - checking for sub-graphs after erosion level '+str(level))
        concept_proposals = self.graph_component_test(G_temp=G_temp,ratio_thresh=ratio_thresh,count_trhesh_low=count_trhesh_low)
        return concept_proposals
    
    def initialize_rand_node_select(self,data):
        self.centroids = {}   
        for i in range(self.k):
            self.centroids[i] = data[i]
            
    def initialize_rand_node_generate(self,data):
        self.centroids = {}   
        mat = np.matrix(data)
        self.golbal_boundaries = list(np.array([np.array(mat.max(0))[0],np.array(mat.min(0))[0]]).T)
        for i in range(self.k):
            self.centroids[i] = np.array([np.random.uniform(x[1],x[0]) for x in self.golbal_boundaries])
    
    def initialize_clusters(self,data,keywords:list=None):
        """
        Make a Pandas DataFrame self.classifications from the 2D data, with empty class and T=0

        Parameters
        ----------
        data : 2D numpy array.
        keywords : list of lists

        """
        self.columns_vector = [str(i) for i in range(data.shape[1])]
        self.columns = ['t','class','kw']+self.columns_vector
        self.classifications = pd.DataFrame(data)
        self.classifications.insert(0,'class',None,0)
        self.classifications.insert(0,'t',None,0)
        self.classifications.insert(0,'kw',None,0)
        self.classifications.columns = self.columns
        self.classifications['kw'] = keywords
        self.classifications['t'] = 0
        self.class_radius = {}
        
        for i in range(self.k):
            self.class_radius[i] = None
   
    def get_distance(self,vec_a,vec_b,distance_metric:str='euclidean'):
        # self.verbose(4,debug='vectors are:'+str(vec_a)+'\n'+str(vec_b))
        if distance_metric == 'euclidean':
            return np.linalg.norm(vec_a-vec_b)
        if distance_metric == 'cosine':
            return spatial.distance.cosine(vec_a,vec_b)
    
    def get_class_min_bounding_box(self,classifications):
        """
        Parameters
        ----------
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t').  provided by self.classifications

        Returns
        -------
        list
            list of minimum bounding boxes for each cluster/class.

        """
        self.ndbbox = {}
        labels = classifications.groupby('class').groups
        for i in labels:
            # i = np.array(classifications[i])
            vecs = classifications[classifications['class']==i][self.columns_vector].values
            try:
                self.ndbbox[i] = np.array([vecs.min(axis=0,keepdims=True)[0],vecs.max(axis=0,keepdims=True)[0]])
            except :
                self.ndbbox[i] = np.zeros((2,i.shape[1]))
                self.verbose(2,warning='Class is empty! returning zero box.')
        return self.ndbbox

    def get_epsilon_radius(self,epsilon_radius:float=None):
        if epsilon_radius==None:
            return min(self.radius.values())*self.boundary_epsilon_coeff
        else:
            return epsilon_radius

    def get_class_radius(self,classifications,centroids,distance_metric:str='euclidean',min_radius:float=None):
        """        
        Parameters
        ----------
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t').  provided by self.classifications
        centroids : dict
            Dict of centroids, provided by self.centroids.
        distance_metric : str, optional
        min_radius : flaot, optional
            A constant value for minimum radius of new-born clusters. (if applicable)
            If None, will automatically use the smallest radius epsilon from the available radius values.
            The default is None.
        Returns
        -------
        list
            list of cluster/class radius.

        """
        self.radius = {}
        labels = classifications.groupby('class').groups
        for i in labels:
            vecs = classifications[classifications['class']==i][self.columns_vector].values
            try:
                centroid = np.array(centroids[i])
                self.radius[i] = max([self.get_distance(vec_a=vector,vec_b=centroid,distance_metric=distance_metric) for vector in vecs])
            except:
                self.radius[i] = self.get_epsilon_radius(epsilon_radius=None)
                self.verbose(2,warning='Exception handled. During radius calculation for class '+str(i)+' an error occuured, so minimum radius was assigned for it.')
        return self.radius
    
    def cluster_neighborhood(self,to_inspect:list,to_ignore:list=None,epsilon:float=5.0):
            """
            Overlaps the bounding boxes to find the neighbouring clusters.
    
            Parameters
            ----------
            to_inspect : list
                list of wanted clusters.
            to_ignore : list of sets, optional
                List of unwanted cluster pairs. The default is None.
            epsilon : float, optional
                The epsilong to grow on each dimension for overlap creation. If>1.0, value will be used as percentage. The default is 0.05.

            Returns
            -------
            List of neighboring cluster as list of pairs.

            """
            pairs = list(itertools.combinations(to_inspect, 2))
            self.get_class_min_bounding_box(self.classifications)
            # self.ndbbox
            self.ndbbox_overlapping = {}
            for c in self.ndbbox.keys():
                if epsilon>=1.0:
                    epsilon_dim = (self.ndbbox[c][1,:]-self.ndbbox[c][0,:])*epsilon/100
                    self.ndbbox_overlapping[c] = np.array([self.ndbbox[c][0,:]-epsilon_dim,self.ndbbox[c][1,:]+epsilon_dim])
                else:
                    self.ndbbox_overlapping[c] = np.array([self.ndbbox[c][0,:]-epsilon,self.ndbbox[c][1,:]+epsilon])
            neighbours = []
            for pair in pairs:
                cond_a = (self.ndbbox_overlapping[pair[0]][0,:]<self.ndbbox_overlapping[pair[1]][1,:]).all()
                cond_b = (self.ndbbox_overlapping[pair[0]][1,:]>self.ndbbox_overlapping[pair[1]][0,:]).all()
    
                if cond_a and cond_b:
                    neighbours.append(set(pair))
            # neighbours = list(set([n for n in neighbours if list(n) not in to_ignore]))
            
            to_ignore = [set(i) for i in to_ignore]
            neighbours_new = []
            for elem in neighbours:
                if elem not in to_ignore:
                    neighbours_new.append(elem)
                    
            self.temp['to_inspect'] = {'data':to_inspect,'t':self.t}
            self.temp['to_ignore'] = {'data':to_ignore,'t':self.t}
            self.temp['neighbours'] = {'data':neighbours_new,'t':self.t}
            
            return [list(i) for i in neighbours_new]
        
    def add_to_clusters(self,data,t,keywords:list=None):
        """
        Update self.classifications using the new 2D data

        Parameters
        ----------
        data : 2D numpy array.
        t : int
            Time-stamp of data (e.g. 0,1,2,3,..,n)
        keywords: list of lists
        
        """
        classifications = pd.DataFrame(data)
        classifications.insert(0,'class',None,0)
        classifications.insert(0,'t',None,0)
        classifications.insert(0,'kw',None,0)
        classifications.columns = self.columns
        classifications['t'] = t
        classifications['kw'] = keywords
        
        self.classifications = self.classifications.append(classifications)
        self.classifications.reset_index(drop=True,inplace=True)
        
    def re_initialize_new_data_clusters(self,t):
        """
        Make a Pandas DataFrame self.classifications from the 2D data, with empty class and T=0
    
        Parameters
        ----------
        data : 2D numpy array.
        t: int
            Classes of time t will be re-set
        """
        self.classifications.loc[self.classifications['t']==t,'class'] = None

        # self.class_radius = {}
        # for i in range(self.k):
        #     self.class_radius[i] = None

    def centroid_stable(self):
        stable = True
        for c in self.centroids:
            try:
                original_centroid = self.centroids_history[-1][c] 
            except KeyError:
                self.verbose(2,debug="New classes and centroids added. considering it a movevement and returning False")
                stable = False
            current_centroid = self.centroids[c]
            self.temp['original_centroids'].append({'t':self.t,'c':c,'original_centroid':original_centroid})
            sums = np.sum((current_centroid-original_centroid)/abs(original_centroid)*100.0)
            self.temp['centroid_sums'].append({'t':self.t,'c':c,'sums':sums})
            movement = abs(sums)
            if movement > self.tol:
                self.verbose(2,debug=str(movement)+' > '+str(self.tol))
                stable = False
            else:
                self.verbose(2,debug=str(movement)+' < '+str(self.tol))
        return stable
    
    def predict(self,data:np.array,distance_metric:str=None):
        """
        Assign data to clusters based on similarity

        Parameters
        ----------
        data : np.array
            2D np.array. array of vectors.
        distance_metric : str, optional
            options are cosine and euclidean. The default is None.

        Returns
        -------
        labels : TYPE
            DESCRIPTION.

        """
        assert len(data.shape)==2, "Incorrect shapes. Expecting a 2D np.array."
        if distance_metric==None:
            distance_metric = self.distance_metric
        labels = list()
        for featureset in data:
            distances = [self.get_distance(featureset,self.centroids[centroid],distance_metric) for centroid in self.centroids]
            labels.append(distances.index(min(distances)))
        return labels
    
    def weight(self,a,t):
        """
        Default weight function: W = 1/((a*time_delta)+1)
        a=-1 means the weight is zero. 

        """
        if a==-1:
            return 0
        
        return 1/((a*t)+1)

    def assign_cluster(self,vec,ignore:list,active:list=None):
        """
        Find the closest centroid to the vector and return the centroid index
    
        Parameters
        ----------
        vector : np.array
        ignore : list, optional
            list of classes to be ignored.
        Returns
        -------
        classification : int
    
        """
        self.temp['vec at assign cluster 650'] = {'data': vec,'t':self.t}
        self.temp['ignore at assign cluster 650'] = {'data': ignore,'t':self.t}
        self.temp['active at assign cluster 650'] = {'data': active,'t':self.t}
        
        distances = {}
        for centroid in self.centroids:
        # for centroid in list(set(self.clasifications[self.classifications['t']==t]['class'].values.tolist())):
            if ignore==None:
                if active==None:
                    distances[centroid] = self.get_distance(vec_a=vec,vec_b=self.centroids[centroid],distance_metric=self.distance_metric) 
                else:
                    if centroid in active:
                        distances[centroid] = self.get_distance(vec_a=vec,vec_b=np.array(list(self.centroids[centroid])),distance_metric=self.distance_metric) 
            else:
                if active==None:
                    if centroid not in ignore:
                        self.temp['centroid at assign cluster 650'] = {'data': self.centroids[centroid],'t':self.t}

                        distances[centroid] = self.get_distance(vec_a=vec,vec_b=np.array(list(self.centroids[centroid])),distance_metric=self.distance_metric) 
                        # distances[centroid] = self.get_distance(vec_a=vec,vec_b=np.array(list(centroid)),distance_metric=self.distance_metric) 

                else:
                    if centroid not in ignore and centroid in active:
                        distances[centroid] = self.get_distance(vec_a=vec,vec_b=self.centroids[centroid],distance_metric=self.distance_metric) 
        
        # if ignore==None:
        #     distances = [self.get_distance(vec,self.centroids[centroid],self.distance_metric) for centroid in self.centroids]
        # else:
        #     if active==None:
        #         distances = [self.get_distance(vec,self.centroids[centroid],self.distance_metric) for centroid in self.centroids if centroid not in ignore]
        #     else:
        #         distances = [self.get_distance(vec,self.centroids[centroid],self.distance_metric) for centroid in self.centroids if centroid not in ignore and centroid in active]
        
        # classification = distances.index(min(distances)) #argmin: get the index of the closest centroid to this featureset/node
        classification = min(distances, key=distances.get)
        return classification
    
    def assign_clusters(self,classifications,ignore:list=None):
        """
        Assign clusters to the list. Not recommended for using on the self.classifications dataframe, as may mix up the indices. If used, make sure to have the indices matched.
        Parameters
        ----------
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t'). 
    
        """
        # for i,featureset in enumerate(data):
        for i,row in classifications[self.columns_vector].iterrows():
            self.classifications['class'][i] = self.assign_cluster(vec=row.values,ignore=ignore)
            
    def assign_clusters_pandas(self,t:int=None,ignore:list=None,active:list=None):
        """
    
        Parameters
        ----------
        classifications: Pandas DataFrame
            Comprises the vector data ('0','1',...,'n'), classification ('class'), and time slice ('t'). 
        t: int
            time slice to assign.
            Set to None to assign all
        ignore : list, optional
            list of classes to be ignored. Default is None.
        """
        if t==None:
            self.classifications['class'] = self.classifications[self.columns_vector].apply(lambda x: self.assign_cluster(vec=x,ignore=ignore),axis = 1)
        else:
            self.classifications.loc[self.classifications['t']==t,'class'] = self.classifications[self.classifications['t']==t][self.columns_vector].apply(lambda x: self.assign_cluster(vec=x,ignore=ignore,active=active),axis = 1)

    def cluster(self,t,n_iter,weights,ignore:list=None):
        """
        Parameters
        ----------
        t : int
            time slice.
        ignore : list, optional
            list of classes to be ignored. Default is None.

        Returns
        -------
        None.

        """
        patience_counter = 0
        for iteration in tqdm(range(n_iter),total=n_iter):
            # Re-initialize clusters
            self.re_initialize_new_data_clusters(t)
            
            # Assign clusters
            self.assign_clusters_pandas(t=self.t,ignore=ignore,active=self.classes_t_start)
            
            # update centroids using time-dependant weighting scheme
            prev_centroids = dict(self.centroids)
            self.centroids_history.append(prev_centroids)
            # for i in self.classifications.groupby('class').groups:
            for i in self.classes_t_start:
                vecs = self.classifications[self.classifications['class']==i][self.columns_vector].values
                vecs = (vecs.T*weights[self.classifications['class']==i].values.T).T
                self.centroids[i] = sum(vecs)/sum(weights[self.classifications['class']==i])
            
            # Compare centroid change to stop iteration
            if self.centroid_stable():
                patience_counter+=1
                if patience_counter>self.patience:
                    self.verbose(1,debug='Centroids are stable within tolerance. Stopping.')
                    break
                self.verbose(2,debug='Centroids are stable within tolerance. Remaining patience:'+str(self.patience-patience_counter))
            else:
                patience_counter=0
    
    def merge_cluster(self,pair:list,t:int,a:float,weight=None):
        self.verbose(2, debug='Getting class='+str(pair))
        # manipulation_classifcations = self.classifications[(self.classifications['class'].isin(list(pair)))]
        
        if self.cumulative:
            manipulation_classifcations = self.classifications[(self.classifications['class'].isin(list(pair)))]
            self.verbose(2, debug='Getting class='+str(pair)+' rows at t=all')
        else:
            manipulation_classifcations = self.classifications[(self.classifications['class'].isin(list(pair))) & (self.classifications['t']==t)]
            self.verbose(2, debug='Getting class='+str(pair)+' rows at t='+str(t))
            
        # get the list of all prior clusters.
        prev_clusters = list(dict(self.centroids).keys())
        # all prior clusters and current clusters are now in ignore list. So the subcluster won't dedicate anythign to them.
        self.verbose(2, debug='Killing the old merging clusters.')
        self.evolution_events[self.evolution_event_counter] = {'t':t,'c':pair[0],'event':'death','location':'merge- 777','timestap':datetime.now()}
        self.evolution_event_counter+=1
        self.evolution_events[self.evolution_event_counter] = {'t':t,'c':pair[1],'event':'death','location':'merge - 779','timestap':datetime.now()}
        self.evolution_event_counter+=1
        
        # assign data to the cluster
        new_cluster_id = max(prev_clusters)+1
        if self.cumulative:
            self.classifications.loc[self.classifications['class'].isin(list(pair)),'class'] = new_cluster_id
            self.verbose(2, debug='Getting class='+str(pair)+' rows at t=all')
        else:
            self.classifications.loc[(self.classifications['class'].isin(list(pair))) & (self.classifications['t']==t),'class'] = new_cluster_id
            self.verbose(2, debug='Getting class='+str(pair)+' rows at t='+str(t))
        self.evolution_events[self.evolution_event_counter] = {'t':t,'c':new_cluster_id,'event':'birth','location':'merge - 808','timestap':datetime.now(),'parents':list(pair)}
        self.evolution_event_counter+=1
        
        # get the new centroid
        delta_t = abs(self.classifications['t']-self.classifications['t'].values.max())
        if weight==None:
            weights = self.weight(a,delta_t)
        else:
            try:
                weights = weight(a,delta_t)
            except:
                self.verbose(0,warning='Exception occuured while trying to get the weights. Please make sure to provide a valid weight function or use default by not providing anything. The function should accept the slope and delta_t inputs. Now will use the default one.')
                weights = self.weight(1,delta_t)
                
        vecs = manipulation_classifcations[self.columns_vector].values
        if self.cumulative:
            vecs = (vecs.T*weights[manipulation_classifcations].values.T).T
            self.centroids[new_cluster_id] = sum(vecs)/sum(weights[manipulation_classifcations])
        else:
            self.centroids[new_cluster_id] = np.average(vecs,axis=0)
        
        # EXPERIMENTAL
        # delete dead centroids
        if self.cumulative:
            for i in pair:
                del self.centroids[i]
            
        # ignore = prev_clusters
        # patience_counter = 0
        # # EXPERIMENTAL SECTION: merge also does a sub clustering and centroid assignment
        # manipulation_classifcations = self.classifications[(self.classifications['class'].isin(list(pair))) & (self.classifications['t']==t)]
        # self.temp['ignore at sub_cluster 786'] = {'data':ignore,'t':self.t}
        # for iteration in tqdm(range(self.n_iter),total=self.n_iter):
        #     self.assign_clusters(classifications=manipulation_classifcations,ignore=ignore)
        #     prev_centroids = dict(self.centroids)
        #     self.centroids_history.append(prev_centroids)
        #     vecs = self.classifications[self.classifications['class']==new_cluster_id][self.columns_vector].values
        #     self.centroids[new_cluster_id] = np.average(vecs,axis=0)
            
        #     # Compare centroid change to stop iteration
        #     if self.centroid_stable():
        #         patience_counter+=1
        #         if patience_counter>self.patience:
        #             self.verbose(2,debug='Centroids are stable within tolerance. Stopping sub-clustering for cluster '+str(new_cluster_id))
        #             break
        #         self.verbose(3,debug='Centroids are stable within tolerance. Remaining patience:'+str(self.patience-patience_counter))
        #     else:
        #         patience_counter=0
        
        
        
        return new_cluster_id
        
    def sub_cluster(self,t:int,to_split:int,new_centroids:list,n_iter:int,a:float,weight=None,sub_k:int=2):
        """
        Parameters
        ----------
        t : int
            time slice.
            
        ignore : list, optional
            list of classes to be ignored. Default is None.
    
        Returns
        -------
        new_cluster_ids: list
            New cluster IDs.
    
        """
        patience_counter = 0
        
        # select rows of the data to manipulate. index will be preserved so it can overwrite the correct rows.
        if self.cumulative:
            manipulation_classifcations = self.classifications[(self.classifications['class']==to_split)]
            self.verbose(2, debug='Getting class='+str(to_split)+' rows at t=all')
        else:
            manipulation_classifcations = self.classifications[(self.classifications['class']==to_split) & (self.classifications['t']==t)]
            self.verbose(2, debug='Getting class='+str(to_split)+' rows at t='+str(t))

        # get the list of all prior clusters.
        prev_clusters = list(dict(self.centroids).keys())
        # all prior clusters and current clusters are now in ignore list. So the subclust won't dedicate anythign to them.
        self.verbose(2, debug='Ignoring previous clusters and killing the old splitting cluster.')
        ignore = prev_clusters
        
        self.evolution_events[self.evolution_event_counter] = {'t':t,'c':to_split,'event':'death','location':'Split - 840','timestap':datetime.now()}
        self.evolution_event_counter+=1
        # create new labels for the new proposed clusters
        new_cluster_ids = list(range(max(prev_clusters)+1,max(prev_clusters)+1+sub_k))
        self.verbose(2, debug='Initializeing centroids for new clusters: '+str(new_cluster_ids))
        # sub-sample the new centroids, so we cut the centrods to the desired number of sub_k
        new_centroids = random.sample(new_centroids,sub_k)
        
        # initializing new centroids, hence new clusters
        for i,c in enumerate(new_cluster_ids):
            self.centroids[c] = new_centroids[i]
            self.evolution_events[self.evolution_event_counter] = {'t':t,'c':c,'event':'birth','location':'Split - 851','timestap':datetime.now(),'parents':[to_split]}
            self.evolution_event_counter+=1
        
        # EXPERIMENTAL
        # delete dead centroids
        if self.cumulative:
            del self.centroids[to_split]

        delta_t = abs(self.classifications['t']-self.classifications['t'].values.max())
        if weight==None:
            weights = self.weight(a,delta_t)
        else:
            try:
                weights = weight(a,delta_t)
            except:
                self.verbose(0,warning='Exception occuured while trying to get the weights. Please make sure to provide a valid weight function or use default by not providing anything. The function should accept the slope and delta_t inputs. Now will use the default one.')
                weights = self.weight(1,delta_t)
                
        self.verbose(2,debug='Starting the iterations...')
        for iteration in tqdm(range(n_iter),total=n_iter):
            # Re-initialize clusters
            # self.re_initialize_new_data_clusters(t)

            # Assign clusters
            self.temp['manipulation_classifcations at sub_cluster 934'] = {'data':manipulation_classifcations,'t':self.t,'timestap':datetime.now()}
            self.temp['ignore at sub_cluster 935'] = {'data':ignore,'t':self.t,'timestap':datetime.now()}
            self.assign_clusters(classifications=manipulation_classifcations,ignore=ignore)
            
            prev_centroids = dict(self.centroids)
            self.centroids_history.append(prev_centroids)
            
            # update centroids using time-dependant weighting scheme
            for i in new_cluster_ids:
                vecs = self.classifications[self.classifications['class']==i][self.columns_vector].values
                if self.cumulative:
                    vecs = (vecs.T*weights[self.classifications['class']==i].values.T).T
                    self.centroids[i] = sum(vecs)/sum(weights[self.classifications['class']==i])
                else:
                    self.centroids[i] = np.average(vecs,axis=0)
            
            # Compare centroid change to stop iteration
            if self.centroid_stable():
                patience_counter+=1
                if patience_counter>self.patience:
                    self.verbose(2,debug='Centroids are stable within tolerance. Stopping sub-clustering for cluster '+str(to_split))
                    break
                self.verbose(3,debug='Centroids are stable within tolerance. Remaining patience:'+str(self.patience-patience_counter))
            else:
                patience_counter=0
                
        return new_cluster_ids

    def fit(self,data,keywords:list=None):
        """
        Perform clustering for T0
        
        REPLACE THIS WITH SKLEARN KMEANS

        Parameters
        ----------
        data : 2D numpy array
            Array of feature arrays.

        """
        # Initialize centroids
        self.evolution_events[self.evolution_event_counter] = {'t':0,'c':list(range(self.k)),'event':'birth','location':'Fit - 894'}
        self.evolution_event_counter+=1
        self.verbose(1,debug='Initializing centroids using method: '+self.initializer)
        if self.initializer=='random_generated':
            self.initialize_rand_node_generate(data)
        elif self.initializer=='random_selected':
            self.initialize_rand_node_select(data)
        self.verbose(1,debug='Initialized centroids')
        
        patience_counter = 0
        for iteration in tqdm(range(self.n_iter),total=self.n_iter):
            # Initialize clusters
            self.initialize_clusters(data,keywords)
            
            # Iterate over data rows and assign clusters
            self.assign_clusters_pandas()
                
            # Update centroids
            prev_centroids = dict(self.centroids)
            self.centroids_history.append(prev_centroids)
            for i in self.classifications.groupby('class').groups:
                vecs = self.classifications[self.classifications['class']==i][self.columns_vector].values
                self.centroids[i] = np.average(vecs,axis=0)
            
            # Compare centroid change to stop iteration
            if self.centroid_stable():
                patience_counter+=1
                if patience_counter>self.patience:
                    self.verbose(1,debug='Centroids are stable within tolerance. Stopping.')
                    break
                self.verbose(2,debug='Centroids are stable within tolerance. Remaining patience:'+str(self.patience-patience_counter))
            else:
                patience_counter=0
        self.classifications_log[self.t] = self.classifications[['kw','class','t']]
        classifications_populations = self.classifications[self.classifications['t']==self.t][['class']].value_counts() # to_ignore is already eliminated in previous step, if any
        classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
        classifications_populations.reset_index(inplace=True)
        self.populations[self.t] = classifications_populations
        self.centroids_history_t.append(self.centroids)
    
    def fit_update(self,additional_data,t,keywords:list=None,n_iter:int=None,weight=None,a:float=None):
        
        """
        Used for updating classifications/clusters for new data. Will use the previous data as weights for centroid handling. 
        You can set a to -1 to remove previous data effectiveness.

        Parameters
        ----------
        additional_data : 2D numpy array.
            Input data.
        t : int
            time sequence number.
        n_iter : int, optional
            Number of iterarions. The default is None.
        weight : method, optional
            The default is automatically calculated using a.
        a : float, optional
            Prior node weight slope. The default is 1.0.
        """
        classifications_populations_old = self.classifications[self.classifications['t']==t-1][['class']].value_counts()
        classifications_populations_old = pd.DataFrame(classifications_populations_old,columns=['population'])
        classifications_populations_old.reset_index(inplace=True)
        
        initial_radius = self.get_class_radius(self.classifications,self.centroids,self.distance_metric)
        self.t= t 
        if n_iter==None:
            n_iter = self.n_iter
        if a==None:
            a = self.a        
        # self.update_assigned_labels = pd.DataFrame([],columns=[i for i in range(additional_data.shape[1])]+['label'])
        self.verbose(1,debug='Updating self.classifications with new data.')        
        self.add_to_clusters(additional_data,t,keywords)
        self.k_fit_update = len(self.classifications[self.classifications['t']==self.t-1]['class'].value_counts().keys())
        self.classes_t_start = list(self.classifications[self.classifications['t']==self.t-1]['class'].value_counts().keys()) # classes alive at this time

        delta_t = abs(self.classifications['t']-self.classifications['t'].values.max())
        if weight==None:
            weights = self.weight(a,delta_t)
        else:
            try:
                weights = weight(a,delta_t)
            except:
                self.verbose(0,warning='Exception occuured while trying to get the weights. Please make sure to provide a valid weight function or use default by not providing anything. The function should accept the slope and delta_t inputs. Now will use the default one.')
                weights = self.weight(1,delta_t)
        # base_k = self.k
        
        
        # Cluster
        self.verbose(1,debug='Initial assignment...')
        self.cluster(t,n_iter,weights)
        
        self.classes_t_start = list(self.classifications[self.classifications['t']==self.t]['class'].value_counts().keys())
        # =============================================================================
        # Death Check
        # =============================================================================
        self.verbose(1,debug='Checking classes for death...')
        new_data_size = additional_data.shape[0]
        expected_cluster_size = new_data_size/(len(self.classes_t_start))
        # cluster population check
        classifications_populations = self.classifications[self.classifications['t']==t][['class']].value_counts()
        classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
        classifications_populations.reset_index(inplace=True)
        
        to_ignore = []
        for i,row in classifications_populations.iterrows():
            if (row['population'] < expected_cluster_size*(self.death_threshold/100)) and (row['population'] < classifications_populations_old['population'][i]*(self.death_threshold/100)):
                # if population is smaller than 10% of the expectation and 10% of the old cluster size at the same time
                to_ignore.append(row['class'])
        
        self.classifications.loc[(self.classifications['t']==t) & (self.classifications['class'].isin(to_ignore)),'class'] = None
        if len(to_ignore) > 0:
            self.verbose(1,debug='Found dead clusters: '+str(to_ignore))
            self.verbose(1,debug='Clustering again with removal of the dead classes')
            self.evolution_events[self.evolution_event_counter] = {'t':t,'c':to_ignore,'event':'death','location':'Death Check - 1001'}
            self.evolution_event_counter +=1
            self.cluster(t,n_iter,weights,to_ignore)
        
        self.verbose(1,debug='Checking classes for death finalized.')
        
        # intera-cluster distances check
        self.verbose(2,debug='Checking classes for radius growth rates')
        new_radius = self.get_class_radius(self.classifications,self.centroids,self.distance_metric)
        # for c in list(set(classifications_populations['class'].values.tolist())): 
        #     try:
        #         old_radius = initial_radius[c]
        #     except :
        #         self.verbose(2,debug=' - Class '+str(c)+' is new and not available in prior data.')
        #         continue
        #     if new_radius[c]/old_radius < 1:
        #         self.verbose(2,debug=' -  - Class '+str(c)+' is shrinking in radius.')
                
        self.classes_t_start = list(self.classifications[self.classifications['t']==self.t]['class'].value_counts().keys())
        
        # =============================================================================
        # Split/Merge Operations
        # =============================================================================
        merge_stable = False
        split_stable = False
        for split_merge_i in tqdm(range(self.merge_split_iters)):
            if merge_stable and split_stable:
                break
                self.verbose(2,debug='Split/Merge is stable; stopping.')
            else:
                merge_stable = False
                split_stable = False
                self.verbose(2,debug='Split/Merge is not stable; going on.')
                # =============================================================================
                # Splitting check
                # =============================================================================
                self.verbose(0,debug='Checking classes for splitting...')
                # cluster population check
                self.verbose(2,debug=' - Checking classes for population proportion growth rates')
                classifications_populations_old = self.classifications[self.classifications['t']==t-1][['class']].value_counts() # we should consider to_ignore aka dead classes for correct total population  at t-1, not <t
                classifications_populations_old = pd.DataFrame(classifications_populations_old,columns=['population'])
                classifications_populations_old.reset_index(inplace=True)
                
                classifications_populations = self.classifications[self.classifications['t']==t][['class']].value_counts() # to_ignore is already eliminated in previous step, if any
                classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
                classifications_populations.reset_index(inplace=True)
                
                # Measure class population growth to other classes
                to_split = {}
                for c in list(set(classifications_populations['class'].values.tolist())):
                    try:
                        class_population_ratio_old = classifications_populations_old[classifications_populations_old['class']==c]['population']/classifications_populations_old['population'].sum()
                    except :
                        self.verbose(2,debug=' -  - Class '+str(c)+' is new and not available in prior data.')
                        continue
                    class_population_ratio = classifications_populations[classifications_populations['class']==c]['population']/classifications_populations['population'].sum()
                    try:
                        growth_rate = class_population_ratio.tolist()[0]/class_population_ratio_old.tolist()[0]
                        to_split[c] = max(0,(growth_rate-self.growth_threshold_population)*10)
                        self.verbose(2,debug=' -  - Class '+str(c)+' got a population growth split vote of '+str(to_split[c]))
                    except:
                        growth_rate = 1
                        self.verbose(2,debug=' -  - Class '+str(c)+': failed to calculate the population growth split vote!')
                        
                # Intera-cluster distances check
                self.verbose(2,debug='Checking classes for radius growth rates')
                
                for c in list(set(classifications_populations['class'].values.tolist())): 
                    try:
                        old_radius = initial_radius[c]
                    except :
                        self.verbose(2,debug=' - Class '+str(c)+' is new and not available in prior data.')
                        continue
                    # if new_radius[c]/old_radius > self.growth_threshold_radius:
                    #     to_split[c] = max(0,(new_radius[c]/old_radius)-self.growth_threshold_radius)
                    #     if c in to_split.keys():
                    #         to_split[c] +=1
                    #     else:
                    #         to_split[c] = 1
                    vote_value = max(0,(new_radius[c]/old_radius)-self.growth_threshold_radius)
                    if c in to_split.keys():
                        to_split[c] +=vote_value
                    else:
                        to_split[c] = vote_value
                    self.verbose(2,debug=' -  - Class '+str(c)+' got a radius growth split vote of '+str(vote_value))
                                    
                # Ontology matching and checking        
                self.verbose(2,debug=' - Checking alive classes at time t='+str(t)+' for splitting by ontologies.')
                self.verbose(2,debug=' -  - vectorizing concepts.')
                self.prepare_ontology()
                
                class_centroid_proposal_concepts = {}
                class_centroid_proposal = {}
                class_concepts = {}
                populations = {}
                for c in list(set(classifications_populations['class'].values.tolist())): 
                    self.verbose(3,debug=' -  - get records in this class.')
                    classifications_c = self.classifications[self.classifications['class']==c]
                    populations[c] = classifications_c.shape[0]
                    
                    self.verbose(3,debug=' -  - matching keywords and concepts.')
                    # try:
                    if len(self.ontology_search_index)>0:
                        self.verbose(3,debug=' -  - using search index instead of search engine.')
                        classifications_c['concepts'] = classifications_c['kw'].progress_apply(lambda x: [self.map_keyword_ontology_from_index(key) for key in x if self.map_keyword_ontology_from_index(key)!=False])
                    # except:
                    else:
                        self.verbose(3,debug=' -  - using search engine. index not available. (It can be very slow!)')
                        classifications_c['concepts'] = classifications_c['kw'].progress_apply(lambda x: [self.map_keyword_ontology(key) for key in x])
                    
                    
                    self.verbose(3,debug=' -  - finding concept roots.')
                    classifications_c['roots'] = classifications_c['concepts'].progress_apply(lambda x: [self.ontology_dict[key]['parents'] for key in x])
                    classifications_c['roots'] = classifications_c['roots'].progress_apply(lambda x: list(chain.from_iterable(x)))
                    
                    
                    self.verbose(3,debug=' -  - generating document per concept ratios.')
                    roots = classifications_c['roots'].values.tolist()
                    flat_roots = list(itertools.chain.from_iterable(roots))
                    concept_counts = pd.Index(flat_roots).value_counts()
                    class_concepts[c] = concept_counts
                    self.temp['concept_counts'] = {'data':concept_counts,'t':t,'c':c}
                    self.temp['classifications_c'] = {'data':classifications_c,'t':t,'c':c}
                    self.temp['roots'] = {'data':roots,'t':t,'c':c}
                                
                    self.verbose(3,debug=' -  - finding classes with multiple concept roots and updating to_split list.')
                    self.verbose(3,debug=' -  -  - generating concept graph in cluster')
                    
                    self.verbose(3,debug=' -  -  - preparing edges')
                    roots = [r for r in roots if len(r)>=1] # docs with at least one root
                    node_names = list(concept_counts.keys())
                    nodes = concept_counts.to_dict()
                    edges = list(itertools.chain.from_iterable([[list(set(list(x))) for x in list(itertools.combinations(list(set(sets)), 2))] for sets in roots])) # make pairs, hence the links
                    
                    self.temp['nodes'] = {'data':nodes,'t':t}
                    self.temp['edges'] = {'data':edges,'t':t}
                    
                    # edges_to_count = [[tuple(edge)] for edge in edges]
                    # keys = list(Counter(itertools.chain(*edges_to_count)).keys())
                    # vals = list(Counter(itertools.chain(*edges_to_count)).values())
                    
                    self.verbose(2,debug=' -  - constructing the concept graph for class '+str(c))
                    G = nx.MultiGraph()
                    G.add_nodes_from(nodes) # nodes with value counts (weights)
                    G.add_edges_from(edges)
                    to_delete = [list(x) for x in list(itertools.combinations(nodes, 2))]
                    self.verbose(3,debug=' -  -  - checking for sub-graphs')
                    concept_proposals = self.graph_component_test(G_temp=G,ratio_thresh=1/self.growth_threshold_population/5,count_trhesh_low=self.death_threshold)
                    self.temp['concept_proposals_first'] = {'data':concept_proposals,'t':t}
                    if len(concept_proposals)>0:
                        self.verbose(2,debug=' -  -  - cluster can be splitted for these concepts as centoids:'+str(concept_proposals))
                        to_split[c] +=self.levels_to_erode
                    else:
                        erode_min = self.levels_to_erode_min
                        erode_max = self.levels_to_erode
                        level = erode_min
                        while erode_max-erode_min>1:
                            level = int((erode_max+erode_min)/2)
                            self.verbose(1,debug=str(erode_min)+'-'+str(level)+'-'+str(erode_max))
                            G_test = copy.deepcopy(G)
                            concept_proposals = self.erosion_component_test(G_tmp=G_test,nodes=nodes,level=level,ratio_thresh=1/10,count_trhesh_low=10)
                            if len(concept_proposals)>0:
                                self.verbose(3,debug=' -  -  - cluster '+str(c)+' can be splitted for these concepts as centoids:'+str(concept_proposals)+' with erosion level '+str(level))
                                erode_max = level
                            else:
                                self.verbose(3,debug=' -  -  - cluster cannot be splitted for class '+str(c)+' with erosion level '+str(level)+' min/max:'+str(erode_min)+'/'+str(erode_max))
                                erode_min = level
                        
                        try:
                            to_split[c] += (self.levels_to_erode-level)
                        except:
                            to_split[c] = (self.levels_to_erode-level)
                        
                        # The old iterative method, instead of binary search
                        # for level in range(self.levels_to_erode_min,self.levels_to_erode):
                        #     G_test = copy.deepcopy(G)
                        #     concept_proposals = self.erosion_component_test(G_tmp=G_test,nodes=nodes,level=level,ratio_thresh=1/10,count_trhesh_low=10)
                        #     if len(concept_proposals)>0:
                        #         self.verbose(3,debug=' -  -  - cluster '+str(c)+' can be splitted for these concepts as centoids:'+str(concept_proposals)+' with erosion level '+str(level))
                        #         try:
                        #             to_split[c] += (self.levels_to_erode-level)
                        #         except:
                        #             to_split[c] = (self.levels_to_erode-level)
                        #         break
                        #     else:
                        #         self.verbose(3,debug=' -  -  - cluster cannot be splitted for class '+str(c)+' with erosion level '+str(level))
                    
                    self.temp['concept_proposals_then'].append({'data':concept_proposals,'t':t})
                    
                    centroid_proposals = []
                    ignores = [] #docs to ignore, as already selected
                    self.verbose(2,debug=' -  - getting centroid proposals.')
                    self.temp['concept_proposals'].append({'concept_proposals':concept_proposals,'t':t,'c':c})
                    for root in list(concept_proposals):
                        self.temp['root'] = {'data':root,'t':t}
                        self.verbose(value=4,debug='getting root vec for: '+root)
                        centroid_proposal,ignore = self.return_root_doc_vec(root=root,classifications_portion=classifications_c,ignores=ignores)
                        centroid_proposals.append(centroid_proposal)
                        ignores.append(ignore)
                    class_centroid_proposal[c] = centroid_proposals
                    class_centroid_proposal_concepts[c] = concept_proposals
                    self.temp['centroid_proposals'].append({'centroid_proposals':centroid_proposals,'t':t})

                class_centroid_proposal = {k:v for k,v in class_centroid_proposal.items() if len(v)>=2}
        
                self.verbose(2,debug=' -  - sub-clustering the records in to_split classes.')
                
                sub_clustered = [] #sub clustered pairs
                
                if class_centroid_proposal==0:
                    split_stable = True
                self.temp['to_split'].append(to_split)
                self.temp['split_populations'].append(populations)
                
                
                if len(class_centroid_proposal)<=0:
                    
                    to_split_auto = [k for k,v in to_split.items() if v>=self.split_auto_threshold]
                    
                    if len(to_split_auto)<=0:
                        to_split_auto = 'None'
                    else:
                        class_centroid_proposal = {}
                        self.verbose(0,debug='No centroids or concepts were proposed for the slpit operation. Attempting to generate random ones...')
                        for c in to_split_auto:
                            
                            class_vecs = self.classifications[self.classifications['class']==c].drop(['t','class','kw'],axis=1).sample(2).values
                            class_centroid_proposal[c] = class_vecs
                            self.verbose(2,debug='Created two class vecs as centroid proposals for class '+str(c))
                        self.temp['class_centroid_proposal'].append(class_centroid_proposal)
                            
                        if self.auto_split == True:
                            user_input = 'a'
                        else:
                            user_input = input("Automatically split clusters: "+str(to_split_auto)+" ?")

                        if user_input=='A' or user_input=='a':
                            for c,v in class_centroid_proposal.items():
                                # print(c,v)
                                self.verbose(1,debug=' -  -  sub clustering cluster '+str(c))
                                to_recluster = int(c)
                                centroid_vecs =  list(class_centroid_proposal[to_recluster])
                                self.temp['centroid_vecs'].append({'centroid_vecs':centroid_vecs,'c':c,'t':t})
                                self.temp['class_centroid_proposal_c'].append({'class_centroid_proposal':class_centroid_proposal[c],'c':c,'t':t})
                                self.verbose(2,debug='reclustering cluster '+str(to_recluster)+' ')
                                sub_clustered.append(self.sub_cluster(t=t, to_split=to_recluster, new_centroids=centroid_vecs,a=a,weight=weight, n_iter=self.n_iter))
                                # del class_centroid_proposal[to_recluster]
                            class_centroid_proposal = {}
                        
                    
                    class_centroid_proposal = {}
                    
                while len(class_centroid_proposal)>0:
                    to_split = {x:(to_split[x]-(populations[x]*self.split_regression_coefficient)+self.split_regression_offset) for x in to_split}
                    to_split = {x:round(to_split[x],2) for x in to_split}
                    print("Cluster split votes are as follows:")
                    print(to_split)

                    to_split_auto = [k for k,v in to_split.items() if v>=self.split_auto_threshold]
                    if len(to_split_auto)<=0:
                        to_split_auto = 'None'
                    
                    if self.auto_split == True:
                        user_input = 'a'
                    else:
                        user_input = input("Which cluster you want to re-cluster? (N: none, A: Auto (Clusters: "+str(to_split_auto)+"), or from: "+str([k for k,v in class_centroid_proposal.items()])+")\n")
                    
                    if user_input=='N' or user_input=='n':
                        class_centroid_proposal = {}
                    elif user_input=='A' or user_input=='a':
                        for c,v in class_centroid_proposal.items():
                            if to_split[c]>=self.split_auto_threshold:
                                self.verbose(1,debug=' -  -  sub clustering cluster '+str(c))
                                to_recluster = int(c)
                                centroid_vecs = class_centroid_proposal[to_recluster]

                                sub_clustered.append(self.sub_cluster(t=t, to_split=to_recluster, new_centroids=centroid_vecs,a=a,weight=weight, n_iter=self.n_iter))
                                # del class_centroid_proposal[to_recluster]
                        class_centroid_proposal = {}    
                    else:
                        cluster_sub_k_input = input("Proposed concepts to split are: "+str(class_centroid_proposal_concepts[int(user_input)])+". How many sub_clusters do you prefer? (N: None and skip sub_clustering, D: Default, or an integer number number)\n")
                        to_recluster = int(user_input)
                        if cluster_sub_k_input!="N" and cluster_sub_k_input!="n":
                            try:
                                self.verbose(1,debug=' -  - sub clustering cluster '+str(user_input))
                                centroid_vecs =  class_centroid_proposal[to_recluster]
                                if cluster_sub_k_input=="D" or cluster_sub_k_input=="d":
                                    sub_clustered.append(self.sub_cluster(t=t, to_split=to_recluster, new_centroids=centroid_vecs,a=a,weight=weight, n_iter=self.n_iter))
                                else:
                                    self.temp['to_recluster at split'] = {'data':to_recluster,'t':self.t}
                                    self.temp['centroid_vecs at split'] = {'data':centroid_vecs,'t':self.t}
                                    self.temp['cluster_sub_k_input at split'] = {'data':cluster_sub_k_input,'t':self.t}
                                    sub_clustered.append(self.sub_cluster(t=t, to_split=min(to_recluster,len(class_centroid_proposal_concepts[int(user_input)])), new_centroids=centroid_vecs,a=a,weight=weight, n_iter=self.n_iter, sub_k = int(cluster_sub_k_input)))
                            except:
                                self.verbose(1,debug=' -  -  input error. Please try again... '+str(user_input))
                        else:
                            self.verbose(1,debug=' -  -  cancelling the sub-clustering operation.')
                            
                        del class_centroid_proposal[to_recluster]
                
                # Update class stats
                self.classes_t_start = list(self.classifications[self.classifications['t']==self.t]['class'].value_counts().keys()) 
                
                # Update class populations
                classifications_populations = self.classifications[self.classifications['t']==t][['class']].value_counts()
                classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
                classifications_populations.reset_index(inplace=True)
                # input("Split ended. Press Enter to continue...")
                
                # =============================================================================
                # Merging check
                # =============================================================================
                self.verbose(0,debug='Checking classes for merging...')
                new_classes = list(set(list(itertools.chain.from_iterable(sub_clustered))))
                all_classes_t = list(set(classifications_populations['class'].values.tolist()))+new_classes
                
                self.verbose(2,debug='Ignoring the new clusters creating in splitting...')
                if not self.merge_sub_clusters:
                    clustered_pairs = [] 
                    
                for subs in sub_clustered :
                    clustered_pairs = clustered_pairs+[set(p) for p in list(itertools.combinations(subs, 2))] # will be used as an ignore list
                
                neighbors = self.cluster_neighborhood(all_classes_t,clustered_pairs,epsilon=self.boundary_epsilon_growth)
                self.verbose(2,debug="Neighbors to construct the pairs are:"+str(neighbors))
                to_merge = []
                
                for pair in neighbors:
                    if len(pair)>1:
                        try:
                            previous_distance = self.get_distance(self.centroids_history_t[-1][pair[0]] ,self.centroids_history_t[-1][pair[1]] ,self.distance_metric)
                            self.verbose(4,debug=" - pair in history. Distance is:"+str(previous_distance))
                        except KeyError:
                            self.verbose(2,debug=" - KeyError pair not in history.")
                            previous_distance = -1000
                        except IndexError:
                            self.verbose(2,debug=" - IndexError pair not in history.")
                            previous_distance = -1000
                        current_distance = self.get_distance(self.centroids[pair[0]] ,self.centroids[pair[1]] ,self.distance_metric)
                        
                        if current_distance<=previous_distance*self.merge_distance_percentage:
                            to_merge.append(pair)
                            self.verbose(2,debug=" - pair are getting closer:"+str(pair))
                        else:
                            self.verbose(4,debug=" - pair are not getting closer:"+str(pair))
                            self.verbose(4,debug=" - distances are:"+str(current_distance)+'>='+str(previous_distance)+'*'+str(self.merge_distance_percentage))
                
                self.verbose(3,debug="The pairs are:"+str(to_merge))
                
                merge_vote = {}
                populations = {}
                # pair_concepts = {}
                for pair_id,pair in enumerate(to_merge):
                    
                    classifications_c = self.classifications[self.classifications['class'].isin(list(pair))]
                    populations[pair_id] = classifications_c.shape[0]
                    self.verbose(2,debug=' -  - finding keywords in concepts.')
                    # try:
                    if len(self.ontology_search_index)>0:
                        self.verbose(2,debug=' -  - using search index instead of search engine.')
                        classifications_c['concepts'] = classifications_c['kw'].progress_apply(lambda x: [self.map_keyword_ontology_from_index(key) for key in x if self.map_keyword_ontology_from_index(key)!=False])
                    # except:
                    else:
                        self.verbose(2,debug=' -  - using search engine. index not available. (It can be very slow!)')
                        classifications_c['concepts'] = classifications_c['kw'].progress_apply(lambda x: [self.map_keyword_ontology(key) for key in x])
                    
                    self.verbose(2,debug=' -  - finding concept roots.')
                    classifications_c['roots'] = classifications_c['concepts'].progress_apply(lambda x: [self.ontology_dict[key]['parents'] for key in x])
                    classifications_c['roots'] = classifications_c['roots'].progress_apply(lambda x: list(chain.from_iterable(x)))
                    
                    # self.verbose(2,debug=' -  - generate document per concept ratios.')
                    roots = classifications_c['roots'].values.tolist() # This paragraph was commented , but do not remember why. 
                    flat_roots = list(itertools.chain.from_iterable(roots))
                    concept_counts = pd.Index(flat_roots).value_counts()
                    # pair_concepts[pair_id] = concept_counts
                                
                    self.verbose(2,debug=' -  - finding pairs with solid concept root components and updating to_merge list.')
                    self.verbose(2,debug=' -  -  - generating concept graph in cluster')
                    
                    self.verbose(2,debug=' -  -  - preparing edges')
                    roots = [r for r in roots if len(r)>1] # docs with at least two roots
                    nodes = list(concept_counts.keys())
                    edges = list(itertools.chain.from_iterable([[list(set(list(x))) for x in list(itertools.combinations(list(set(sets)), 2))] for sets in roots])) # make pairs, hence the links
        
                    # edges_to_count = [[tuple(edge)] for edge in edges]
                    # keys = list(Counter(itertools.chain(*edges_to_count)).keys())
                    # vals = list(Counter(itertools.chain(*edges_to_count)).values())
                    
                    self.verbose(2,debug=' -  -  - constructing the graph')
                    G = nx.MultiGraph()
                    G.add_nodes_from(nodes)
                    G.add_edges_from(edges)

                    erode_min = self.levels_to_erode_min
                    erode_max = self.levels_to_erode
                    level = erode_min
                    while erode_max-erode_min>1:
                        level = int((erode_max+erode_min)/2)
                        self.verbose(1,debug=str(erode_min)+'-'+str(level)+'-'+str(erode_max))
                        G_test = copy.deepcopy(G)
                        concept_proposals = self.erosion_component_test(G_tmp=G_test,nodes=nodes,level=level,ratio_thresh=1/10,count_trhesh_low=10)
                        if len(concept_proposals)>0:
                            self.verbose(3,debug=' -  -  - pair '+str(pair_id)+' can be splitted for these concepts as centoids:'+str(concept_proposals)+' with erosion level '+str(level))
                            erode_max = level
                        else:
                            self.verbose(3,debug=' -  -  - cluster cannot be splitted for class '+str(c)+' with erosion level '+str(level)+' min/max:'+str(erode_min)+'/'+str(erode_max))
                            erode_min = level
                    
                    try:
                        merge_vote[pair_id] += level
                    except:
                        merge_vote[pair_id] = level
                        
                
                if len(merge_vote)==0:
                    merge_stable = True
                

                merged = [] #sub clustered pairs
                merge_vote = {k:(merge_vote[k]-(populations[k]*self.merge_regression_coefficient)+self.merge_regression_offset) for k,v in merge_vote.items()}
                try: 
                    self.temp['merge_populations'].append({'t':t,'populations':populations})
                except:
                    self.temp['merge_populations'] = [{'t':t,'populations':populations}]
                    
                try:
                    self.temp['merge_votes'].append({'t':t,'merge_vote':merge_vote,'to_merge':to_merge})
                except:
                    self.temp['merge_votes'] = [{'t':t,'merge_vote':merge_vote,'to_merge':to_merge}]
                    
                # If a multiple merges with shared clusters across them is needed, 
                #      find out the cluster of clusters to be merged, then only merged the top wanted of the bunch (component)
                
                merge_vote_auto = [to_merge[k] for k,v in merge_vote.items() if v>=self.merge_auto_threshold]
                merge_vote_graph = nx.Graph()
                merge_vote_graph.add_edges_from(merge_vote_auto)
                tgl = [len(c) for c in sorted(nx.connected_components(merge_vote_graph), key=len, reverse=True)]# number of clusters within the connected components
                tgc = [c for c in sorted(nx.connected_components(merge_vote_graph), key=len, reverse=True)] # clusters within the connected components
                merge_vote_sorted = dict(sorted(merge_vote.items(), key=lambda item: item[1], reverse=True))
                merge_vote_keys = list(merge_vote_sorted.keys())
                # if len(tgl)<len(list(tg.nodes))/5:
                best_merge = [] 
                for enum_i,enum_comp in enumerate(tgc):
                    for enum_j,enum_pair in enumerate(to_merge):
                        candidate = to_merge[merge_vote_keys[enum_j]]
                        if (candidate[0] in enum_comp) or (candidate[1] in enum_comp):
                            best_merge.append(candidate)
                            break
                         
                    self.verbose(1,debug=' -  -  too many to merge. Will use the max only. Was expecting at least more than '+str(len(list(merge_vote_graph.nodes))/5)+' connected components/repeated node in pairs, but got only '+str(len(tgl))+' cluster of clusters wanting to merge. Will only merge: '+str(best_merge))
                    merge_vote_auto = best_merge.copy()
                    
                if len(merge_vote_auto)<=0:
                    merge_vote_auto = 'None'
                    
                # self.verbose(0,debug="Voting outcome for merging is: "+str([[to_merge[k],v] for k,v in merge_vote.items()]))
                print("Voting outcome for merging is: "+str([[to_merge[k],v] for k,v in merge_vote.items()]))
                while len(merge_vote)>0:
                    if self.auto_merge == True:
                        user_input = 'a'
                    else:
                        user_input = input("Which pair you want to merge? (N: none, A: Auto(Clusters:"+str(merge_vote_auto)+"), or select id of the pair from: "+str([{k:to_merge[k]} for k,v in merge_vote.items()])+")\n")
                    if user_input=='N' or user_input=='n':
                        merge_vote = {}
                    elif user_input=='A' or user_input=='a':
                        if merge_vote_auto!='None':
                            for k in merge_vote_auto:
                                self.verbose(1,debug=' -  -  merging '+str(k))
                                merged.append(self.merge_cluster(k,t,a=a,weight=weight))
                        merge_vote = {}
                    else:
                        try:
                            self.verbose(1,debug=' -  -  merging '+str(to_merge[int(user_input)]))
                            merged.append(self.merge_cluster(to_merge[int(user_input)],t,a=a,weight=weight))
                            del merge_vote[int(user_input)]
                            del to_merge[int(user_input)]
                        except:
                            self.verbose(1,debug=' -  -  input error. Please try again... '+str(user_input))
                
                # Save a copy as log for labeling
                self.classifications_log[self.t] = self.classifications[['kw','class','t']].copy()
                self.classes_t_start = list(self.classifications[self.classifications['t']==self.t]['class']
                                            .value_counts().keys())
                
                
                classifications_populations = self.classifications[self.classifications['t']==self.t][['class']].value_counts() # to_ignore is already eliminated in previous step, if any
                classifications_populations = pd.DataFrame(classifications_populations,columns=['population'])
                classifications_populations.reset_index(inplace=True)
                self.populations[self.t] = classifications_populations
        self.centroids_history_t.append(self.centroids)
                # input("Merge ended. Press Enter to continue...")

# SCOPUS DATA -- Doc2Vec
# =============================================================================
# Load data and init
# =============================================================================
def main(args,date):
    # kw_num = 6 # max number of keywords per doc
    # datapath = '/home/sahand/GoogleDrive/sohanad1990/Data/' 

    # classifications_path = 'Corpus/Scopus new/clean/clustering results/ESWA/'
    # model_path = 'FastText Models/gensim41/FastText100D-dim-scopus-update-gensim41-w5.model'
    # ontology_path = 'Corpus/Taxonomy/concept_parents lvl2 DFS'
    # ontology_indexed_path = 'Corpus/Scopus new/clean/kw ontology search/keyword_search_pre-index.json'
    # all_columns_path = 'Corpus/Scopus new/clean/keyword pre-processed for fasttext - nov14'
    # all_column_years_path = 'Corpus/Scopus new/clean/data with abstract'
    # concept_column_path = 'Corpus/Scopus new/clean/mapped concepts for keywords'
    # concept_embeddings_path = 'Corpus/Scopus new/embeddings/concepts/node2vec/50D p1 q05 len20 average of concepts'
    # corpus_path = 'Corpus/Scopus new/clean/abstract_title method_b_3'
    # vectros_main_path = 'Corpus/Scopus new/embeddings/doc2vec 100D dm=1 window=12 gensim41'

    kw_num = args.kw_num # max number of keywords per doc
    datapath = args.datapath #'/home/sahand/GoogleDrive/sohanad1990/Data/'

    classifications_path = args.classifications_path #'Corpus/Scopus new/clean/clustering results/ESWA/'
    model_path = args.model_path #'FastText Models/gensim41/FastText100D-dim-scopus-update-gensim41-w5.model'
    ontology_path = args.ontology_path #'Corpus/Taxonomy/concept_parents lvl2 DFS'
    ontology_indexed_path = args.ontology_indexed_path #'Corpus/Scopus new/clean/kw ontology search/keyword_search_pre-index.json'
    all_columns_path = args.all_columns_path #'Corpus/Scopus new/clean/keyword pre-processed for fasttext - nov14'
    all_column_years_path = args.all_column_years_path #'Corpus/Scopus new/clean/data with abstract'
    concept_column_path = args.concept_column_path #'Corpus/Scopus new/clean/mapped concepts for keywords'
    concept_embeddings_path = args.concept_embeddings_path #'Corpus/Scopus new/embeddings/concepts/node2vec/50D p1 q05 len20 average of concepts'
    corpus_path = args.corpus_path #'Corpus/Scopus new/clean/abstract_title method_b_3'
    vectros_main_path = args.vectros_main_path #'Corpus/Scopus new/embeddings/doc2vec 100D dm=1 window=12 gensim41'
    # datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/' #C1314
    gensim_model_address = datapath+model_path
    #'Doc2Vec Models/b3-gensim41/dim-scop doc2vec 300D dm=1 window=10 b3 gensim41'
    model_AI = fasttext_gensim.load(gensim_model_address)
    # model_AI.save(datapath+'Corpus/Dimensions All/models/Fasttext/FastText100D-dim-scopus-update-gensim383.model')

    k0 = args.n_clusters_init # 7 # number of clusters
    skip_years = args.skip_years # 2001 # skip the years before this year inclusively
    t_zero = args.t_zero # 2000 # the year that t_0 or the first chuck of clustering data is up to
    t_max = args.t_max # 2021 # the last year to end the clustering

    # =============================================================================
    # Read data
    with open(datapath+ontology_path) as f:
        ontology_table = json.load(f)

    with open(datapath+ontology_indexed_path) as f:
        ont_index = json.load(f)

    all_columns = pd.read_csv(datapath+all_columns_path)[['id','eid','DE']] # use the pre-processed keyword data as the main source
    all_columns_ = pd.read_csv(datapath+all_column_years_path)[['PY']] # use data_with_abstract to get the years
    concept_column = pd.read_csv(datapath+concept_column_path)
    concept_embeddings = pd.read_csv(datapath+concept_embeddings_path).drop('Unnamed: 0',axis=1,inplace=False)
    concept_column.concepts = concept_column.concepts.str.split('|')
    all_columns['PY'] = all_columns_['PY']
    all_columns['concepts'] = concept_column.concepts

    # all_columns[pd.isna(all_columns['concepts'])]

    del all_columns_
    gc.collect()

    corpus_data = pd.read_csv(datapath+corpus_path)[['abstract','id']] # get the pre-processed abstracts
    corpus_data.columns = ['abstract','id_n']
    vectors_main = pd.read_csv(datapath+vectros_main_path)#Corpus/Scopus new/embeddings/doc2vec 300D dm=1 window=10 b3 gensim41
    # vectors_main = pd.read_csv(datapath+'Corpus/Scopus new/embeddings/node2vec100D p2 q1 len5-w50 mc1 mt_w2v unsorted.csv',index_col=0).reset_index()#Corpus/Scopus new/embeddings/doc2vec 300D dm=1 window=10 b3 gensim41

    print('Data loaded.')

    # =============================================================================
    # Check data
    if all_columns.shape[0]!=corpus_data.shape[0]:
        print('Oh no! data mismatch here...Please fix it!')

    if all_columns.shape[0]!=vectors_main.shape[0]:
        print('Oh no! data mismatch here...Please fix it!')

    if all_columns.shape[0]!=concept_embeddings.shape[0]:
        print('Oh no! data mismatch here...Please fix it!')

    if all_columns.shape[0]!=concept_column.shape[0]:
        print('Oh no! data mismatch here...Please fix it!')

    print('Data checked.')

    print("all_columns",all_columns)
    print("corpus_data",corpus_data)
    print("vectors_main",vectors_main)
    print("concept_embeddings",concept_embeddings)
    print("concept_column",concept_column)

    # =============================================================================
    # Concat data
    combined_data = pd.concat([all_columns, corpus_data, vectors_main,concept_embeddings], axis=1)
    combined_data = combined_data.dropna(subset=['id','PY','abstract'])

    # =============================================================================
    # Clean data
    combined_data = combined_data.drop_duplicates(subset='id', keep="last")
    combined_data.id.value_counts()
    combined_data.loc[pd.isna(combined_data['DE']),'DE'] = ''
    combined_data = combined_data[pd.notna(combined_data['concepts'])]

    # check for consistency in the combined data
    assert combined_data['id'].equals(combined_data['id_n']), "Oh no! id mismatch here...self.centroids[i] = np.average(vecs,axis=0) Please fix it!"


    # combined_data.PY.hist(bins=60)

    # =============================================================================
    # prepare keywords
    combined_data['DE'] = combined_data['DE'].progress_apply(lambda x: x.split(" | "))
    combined_data['DE'] = combined_data['DE'].progress_apply(lambda x: x[:kw_num] if len(x)>kw_num-1 else x) # cut off to 6 keywords only
    # vectors_main.drop('id',axis=1,inplace=True)

    # =============================================================================
    # intialize the model
    model = OGC(v=5,
                k=k0,
                max_keyword_distance=args.max_keyword_distance,
                boundary_epsilon_growth=args.boundary_epsilon_growth,
                distance_metric=args.distance_metric,
                cumulative=args.cumulative,
                merge_split_iters=args.merge_split_iters,
                split_auto_threshold=args.split_auto_threshold,
                merge_auto_threshold=args.merge_auto_threshold,
                split_regression_coefficient=args.split_regression_coefficient,
                split_regression_offset=args.split_regression_offset,
                merge_regression_coefficient=args.merge_regression_coefficient,
                merge_regression_offset=args.merge_regression_offset,
                initializer=args.initializer,
                log=args.log) 
    model.set_ontology_dict(ontology_table)
    model.set_keyword_embedding_model(model_AI)
    model.set_ontology_keyword_search_index(ont_index)
    ontology_dict = model.prepare_ontology()

    vectors_t0 = combined_data[(combined_data.PY<t_zero)] # & (combined_data.PY>1999)
    vectors_t0_full = vectors_t0.copy()
    keywords = vectors_t0['DE'].values.tolist()
    vectors_t0.drop(['eid','PY','id','abstract','DE','id_n','concepts'],axis=1,inplace=True)
    vectors_t0_fixed = vectors_t0.copy()
    # vectors_t0_full = combined_data.loc[vectors_t0.index]

    # run the model for the first time slice, then get the clusters
    # dendrogram = aa.fancy_dendrogram(sch.linkage(vectors_t0, method='ward'),truncate_mode='lastp',p=800,show_contracted=True,figsize=(15,9)) #single #average #ward
    model.fit(vectors_t0.values,keywords) #merge_sub_clusters=True,merge_split_iters=20

    # =============================================================================
    # Evaluation of time slice 0

    # predicted_labels = model.predict(vectors_t0.values)
    # classifications = model.classifications
    # pd.DataFrame(predicted_labels).hist(bins=k0)
    # gs = pd.DataFrame(predicted_labels,columns=['labels']).value_counts()
    # dbi_0 = davies_bouldin_score(vectors_t0.values, predicted_labels)
    # sil_0 = silhouette_score(vectors_t0.values, predicted_labels)
    # chs_0 = calinski_harabasz_score(vectors_t0.values,predicted_labels)
    # =============================================================================
    global model_backup
    model_backup = copy.deepcopy(model)
    # model_backup_initial = copy.deepcopy(model)

    tmps = []
    logs = []
    # classifications_hist = []
    # classifications_log = []
    # tmps.append(copy.deepcopy(model_backup).temp)
    # logs.append(copy.deepcopy(model_backup).event_log)

    # model = copy.deepcopy(model_backup)
    model.v=5
    model.log = True
    print("verbose",model.v)
    # model.merge_split_iters = args.merge_split_iters

    for t,year in enumerate(range(t_zero,t_max)):
        if year>skip_years:
            print('processing year',t+1,year)
            model.v=5
            vectors_t1 = combined_data[combined_data.PY==year]
            keywords = vectors_t1['DE'].values.tolist()
            vectors_t1.drop(['eid','PY','id','abstract','DE','id_n','concepts'],axis=1,inplace=True)
            model.fit_update(vectors_t1.values, t+1, keywords)
            # model.get_class_radius(model.classifications,model.centroids,model.distance_metric)
            # model.get_class_radius(model.classifications,model.centroids,model.distance_metric)
            tmps.append(copy.deepcopy(model).temp)
            logs.append(copy.deepcopy(model).event_log) 
            # classifications_hist.append(copy.deepcopy(model)) 
            # classifications_log.append(copy.deepcopy(model).classifications_log) 
            model_backup = copy.deepcopy(model)

    # =============================================================================
    # Evaluation of time slices
    # =============================================================================
    try:
        if args.evaluate_model:
            dbi = []
            sil = []
            chs = []

            for y in tqdm(range(args.eval_start,args.eval_end,args.eval_step)):
                vectors_t2 = combined_data[combined_data.PY==y]
                keywords = vectors_t2['DE'].values.tolist()
                vectors_t2.drop(['eid','PY','id','abstract','DE','id_n','concepts'],axis=1,inplace=True)
                predicted_labels_1 = model.predict(vectors_t2.values)

                dbi.append(davies_bouldin_score(vectors_t2.values, predicted_labels_1))
                sil.append(silhouette_score(vectors_t2.values, predicted_labels_1))
                chs.append(calinski_harabasz_score(vectors_t2.values,predicted_labels_1))

            scores = pd.DataFrame({'dbi':dbi,'sil':sil,'chs':chs})
            scores['year'] = list(range(args.eval_start,args.eval_end,args.eval_step))
            scores.to_csv(datapath+classifications_path+date+' scores.csv',index=False)
    except Exception as e:
        print("Evaluation failed.")
        print(e)

    # =============================================================================
    # Save model and classifications
    # temp = model.temp['root_selection_return_fail']['data']['classifications_portion']['roots']
    classifications = model.classifications
    date = datetime.today().strftime('%Y-%m-%d')
    if args.save_classifications:
        classifications.to_csv(datapath+classifications_path+date+' results.csv',index=False)

    if args.save_model:
        with open(datapath+classifications_path+date+' model.pkl', 'wb') as outp:
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    return model, combined_data

def visualise(args,model,date):

    datapath = args.datapath
    # Save the evolutions
    
    evolutions = model.evolution_events
    evolutions_new = {}
    for k,v in evolutions.items():
        evolutions_new[k] = {}
        evolutions_new[k]['c'] = evolutions[k]['c']
        evolutions_new[k]['event'] = evolutions[k]['event']
        evolutions_new[k]['location'] = evolutions[k]['location']
        evolutions_new[k]['t'] = evolutions[k]['t']
    with open(datapath+args.clustering_subdir+date+' evolutions.json', "w") as outfile:
        json.dump(evolutions_new, outfile)
    model.populations

    G = nx.DiGraph()
    t_prev = 0
    clusters = []
    last_died = []
    last_died_t = None
    last_event = None
    clusters_hist = []
    new_nodes = {}
    delayed_kills = []
    t = 0
    for i,evo in evolutions.items():
        if t!= evo['t']:
            # Time has changed
            new_t = True
            new_nodes = {}
            # [G.remove_node(cluster) for cluster in delayed_kills]
            # delayed_kills = []
        else:
            new_t = False
            
        t = evo['t']
        c = evo['c']
        e = evo['event']
        l = evo['location'].split(' ')[0].lower()
        
        print(i,t,e,c)
        # if i>74:
        #     print(clusters)
        #     input('press enter to continue')
        if t-t_prev>0:
            forgotten_edges = []
            for j in range (t_prev,t):
                # clusters_int = [int(k.split('-')[0]) for k in clusters]
                forgotten_edges = forgotten_edges+[[str(k)+'-t'+str(j),str(k)+'-t'+str(j+1)] for k in clusters]
            G.add_edges_from(forgotten_edges)

        if e=='birth':
            try:
                if len(c)>0: #only happens at start
                    clusters = clusters+c
            except:
                clusters.append(c)
                for k in last_died:
                    child = str(c)+'-t'+str(t)
                    if evo['location'].split(' - ')[0]=='merge':
                        died_node = str(k)+'-t'+str(t)
                        father = [k for k,v in new_nodes.items() if died_node in v]
                        # print(father,died_node,new_nodes.items())
                        if len(father)>0:
                            G.add_edges_from([[died_node,child]])
                            print('> NE',died_node,child)
                        else:
                            G.add_edges_from([[str(k)+'-t'+str(t-1),child]])
                    else:
                        G.add_edges_from([[str(k)+'-t'+str(t-1),child]])
                    
                    try:
                        new_nodes[str(k)+'-t'+str(t-1)].append(child)
                    except:
                        new_nodes[str(k)+'-t'+str(t-1)] = [child]
        if e=='death':
            if (last_died_t!=t or last_event=='birth'):
                last_died = []
            last_died.append(c)
            last_died_t = t
            clusters.remove(c)
            # if str(c)+'-c'+str(t) in new_nodes:
            #     # delay the kill
            #     delayed_kills.append(str(c)+'-c'+str(t))
            # else:
            #     G.remove_node(str(c)+'-c'+str(t))
            deleted_node = str(c)+'-t'+str(t)
            G.remove_node(deleted_node)
            father = [k for k,v in new_nodes.items() if deleted_node in v]
            if len(father)>0:
                print('> NED',father[0],deleted_node)
                G.add_edge(father[0],deleted_node)
        
        clusters_hist.append(clusters.copy())
        last_event = e
        t_prev = t

    print(G.nodes)

    plt.title('Evolution Map')
    plt.figure(figsize=(22,12))
    pos =graphviz_layout(G, prog='dot')
    A = nx.nx_agraph.to_agraph(G)
    # A.node_attr["shape"] = 'ellipse'
    # A.node_attr["color"] = 'black'
    # A.edge_attr["color"] = 'red'
    colors = []
    for i, node in enumerate(A.iternodes()):
        if G.degree(node)==0:
            # A.node_attr["color"] = 'black'
            colors.append('green')
        elif G.degree(node)==1:
            # A.node_attr["color"] = 'blue'
            colors.append('yellow')
        elif G.degree(node)==3:
            # print('evolution found')
            # A.node_attr.update(color="red")
            colors.append('pink')
        else:
            colors.append('cyan')

    # A.layout(prog="dot")
    # A.draw('out.svg')
    nx.draw(A, pos, node_color = colors, with_labels=True, arrows=True, node_size = 350)
    plt.savefig(datapath+args.clustering_path+date+' evolution map colored.svg')

# r = {k:v for k,v in tqdm(evolutions.items()) if v['c'] == 62}
# nx.draw(G, with_labels=True)

def labeling(args,model):
    stop_words = set(stopwords.words("english"))
    datapath = args.datapath

    classifications = model.classifications_log
    # classifications = classifications_log[-1]
    for t in range(22): # Limited keywords to time t
        # t  = 4
        # TF-IDF (CTF-ICF)
        cluster_as_string = []
        cluster_ids = []
        year_kw = classifications[t]['kw'].apply(lambda x: kw_to_string(x))
        clusters_df = classifications[t].copy()
        clusters = clusters_df.groupby('class').groups
        for key in clusters.keys():
            cluster_as_string.append(' '.join(year_kw[list(clusters[key])]))
            cluster_ids.append(key)
        cluster_keywords_tfidf = get_abstract_keywords(cluster_as_string,1000,max_df=0.8)
        
        cluster_keywords = []
        cluster_index = 0
        for i,items in enumerate(cluster_keywords_tfidf):
            c = cluster_ids[i]
            items_tmp = [c]
            for item in items:
                max_data = find_max_item_value_in_all_cluster(cluster_keywords_tfidf,item,cluster_index)
                items_tmp.append(item+' ('+str(items[item])+' | '+str(max_data[0])+'/'+str(max_data[1])+')') # (item+' :'+str(items[item])+' / '+str( max of item in all other rows))
            cluster_keywords.append(items_tmp)
            
            cluster_index+=1
        cluster_keywords_df = pd.DataFrame(cluster_keywords)
        cluster_keywords_df.to_csv(datapath+args.clustering_path+'t'+str(t)+' labels c.csv',index=False,header=False)
        
        
        # Get term cluster labels (just terms and not scores)
        cluster_keywords_terms = []
        cluster_keywords_scores = []
        for item in cluster_keywords_tfidf:
            cluster_keywords_terms.append(list(item.keys()))
            cluster_keywords_scores.append(list(item.values()))
        
        pd.DataFrame(cluster_keywords_terms).T.to_csv(datapath+args.clustering_path+'t'+str(t)+' terms.csv',index=False)
        pd.DataFrame(cluster_keywords_scores).T.to_csv(datapath+args.clustering_path+'t'+str(t)+' scores.csv',index=False)
        
        # Get term frequencies for each period
        terms = ' '.join(cluster_as_string).split()
        terms = [x for x in terms if x not in list(stop_words)]
        pd.DataFrame(terms,columns=['terms'])['terms'].value_counts().to_csv(datapath+args.clustering_path+'t'+str(t)+' frequency.csv',header=False)

    final_classifications = classifications[21]
    year_clusters = final_classifications[final_classifications['t']==21]['class'].value_counts().reset_index()
    year_clusters.columns = ['c','count']

    all_nodes = list(G.nodes)
    all_nodes = pd.DataFrame([[int(x.split('-t')[0]),int(x.split('-t')[1])] for x in all_nodes],columns=['c','t'])
    all_nodes_g_max = all_nodes.groupby('c').max()
    all_nodes_g_min = all_nodes.groupby('c').min()
    all_nodes_g = all_nodes_g_min.copy()
    all_nodes_g['tm'] = all_nodes_g_max['t']

    for t in range(22): # All keywords until time t
        # t  = 4
        # TF-IDF (CTF-ICF)
        final_classifications = classifications[t]
        year_clusters = final_classifications[final_classifications['t']==t]['class'].value_counts()
        year_clusters = list(year_clusters.index)
        cluster_as_string = []
        cluster_ids = []
        
        working_df = classifications[t][classifications[t]['class'].isin(year_clusters)]
        
        year_kw = working_df['kw'].apply(lambda x: kw_to_string(x))
        clusters_df = working_df.copy()
        clusters = clusters_df.groupby('class').groups
        for key in clusters.keys():
            cluster_as_string.append(' '.join(year_kw[list(clusters[key])]))
            cluster_ids.append(key)
        cluster_keywords_tfidf = get_abstract_keywords(cluster_as_string,1000,max_df=0.8)
        
        cluster_keywords = []
        cluster_index = 0
        for i,items in enumerate(cluster_keywords_tfidf):
            c = cluster_ids[i]
            items_tmp = [c]
            for item in items:
                max_data = find_max_item_value_in_all_cluster(cluster_keywords_tfidf,item,cluster_index)
                items_tmp.append(item+' ('+str(items[item])+' | '+str(max_data[0])+'/'+str(max_data[1])+')') # (item+' :'+str(items[item])+' / '+str( max of item in all other rows))
            cluster_keywords.append(items_tmp)
            
            cluster_index+=1
        cluster_keywords_df = pd.DataFrame(cluster_keywords)
        cluster_keywords_df.to_csv(datapath+args.clustering_path+'t'+str(t)+' labels c t.csv',index=False,header=False)
        
def benchmark_sample(model):
    classifications_pure = model.classifications
    # cluster_populations = classifications_pure['class'].value_counts()
    classifications_t = classifications_pure[classifications_pure['t']>=21]
    labels = classifications_t['class']
    vecs = classifications_t.drop(['class','t','kw'],axis=1,inplace=False)

    dbi = davies_bouldin_score(vecs.values, labels)
    sil = silhouette_score(vecs.values, labels)
    chs = calinski_harabasz_score(vecs.values,labels)

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='OGC clustering')
    parser.add_argument('--datapath', type=str, default='/home/sahand/GoogleDrive/sohanad1990/Data/', help='Data path')
    parser.add_argument('--model_path', type=str, default='embeddings/fasttext gensim41/FastText100D-dim-scopus-update-gensim41-w5.model', help='Model path')
    parser.add_argument('--ontology_path', type=str, default='Corpus/ontology/concept_parents lvl2 DFS', help='Ontology path')
    parser.add_argument('--ontology_indexed_path', type=str, default='Corpus/ontology/keyword_search_pre-index.json', help='Ontology indexed path')
    parser.add_argument('--all_columns_path', type=str, default='Corpus/Scopus/keyword pre-processed for fasttext - nov14', help='All columns path')
    parser.add_argument('--all_column_years_path', type=str, default='Corpus/Scopus/data with abstract', help='All column data with years column path')
    parser.add_argument('--concept_column_path', type=str, default='Corpus/Scopus/mapped concepts for keywords', help='Concept column path')
    parser.add_argument('--concept_embeddings_path', type=str, default='embeddings/node2vec 50D p1 q05 len20 average of concepts', help='Concept embeddings path')
    parser.add_argument('--corpus_path', type=str, default='Corpus/Scopus/abstract_title method_b_3', help='Corpus path')
    parser.add_argument('--vectros_main_path', type=str, default='embeddings/doc2vec 100D dm=1 window=12 gensim41', help='Vectros main path')
    parser.add_argument('--clustering_path', type=str, default='clustering results/ESWA/', help='Clustering path')
    parser.add_argument('--classifications_path', type=str, default='clustering results/ESWA/', help='Classification path')
    parser.add_argument('--kw_num', type=int, default=6, help='Max number of keywords per doc')
    parser.add_argument('--n_clusters_init', type=int, default=7, help='Number of clusters')
    parser.add_argument('--skip_years', type=int, default=0, help='Skip the years before this year inclusively')
    parser.add_argument('--t_zero', type=int, default=2000, help='The year that t_0 or the first chuck of clustering data is up to')
    parser.add_argument('--t_max', type=int, default=2021, help='The last year to end the clustering')
    parser.add_argument('--evaluate_model', type=bool, default=True, help='Evaluate model')
    parser.add_argument('--eval_start', type=int, default=2001, help='Evaluation start year')
    parser.add_argument('--eval_end', type=int, default=2021, help='Evaluation end year')
    parser.add_argument('--eval_step', type=int, default=1, help='Evaluation step')
    parser.add_argument('--save_classifications', type=bool, default=True, help='Save classifications')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--verbose', type=int, default=5, help='Verbose')
    parser.add_argument('--merge_sub_clusters', type=bool, default=True, help='Merge sub clusters')

    parser.add_argument('--max_keyword_distance', type=float, default=0.55, help='Max keyword distance')
    parser.add_argument('--boundary_epsilon_growth', type=float, default=0, help='Boundary epsilon growth')
    parser.add_argument('--distance_metric', type=str, default='cosine', help='Distance metric')
    parser.add_argument('--cumulative', type=bool, default=False, help='Cumulative')
    parser.add_argument('--log', type=bool, default=True, help='Log')

    parser.add_argument('--merge_split_iters', type=int, default=1, help='Merge split iters')
    parser.add_argument('--merge_auto_threshold', type=int, default=2100, help='Merge auto threshold')
    parser.add_argument('--split_auto_threshold', type=int, default=2450, help='Split auto threshold')

    parser.add_argument('--merge_regression_coefficient', type=float, default=0.7047367793518765, help='Merge regression coefficient')
    parser.add_argument('--merge_regression_offset', type=float, default=2000, help='Merge regression offset')
    parser.add_argument('--split_regression_coefficient', type=float, default=-0.7147367793518765, help='Split regression coefficient')
    parser.add_argument('--split_regression_offset', type=float, default=-48000, help='Split regression offset')
    parser.add_argument('--initializer', type=str, default='random_generated', help='Initializer')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_arguments()
    date = str(datetime.today().strftime('%Y-%m-%d'))
    print('Running model...')
    model, combined_data = main(args,date)
    print('Running visualisation...')
    visualise(args,model,date)
    print('Running labeling...')
    labeling(args,model)

