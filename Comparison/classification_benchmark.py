#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:10:03 2020

@author: sahand
"""

import sys
import time
import gc
import os
from pathlib import Path
import numpy as np
import pandas as pd
import statistics as st
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import pyplot
from random import randint

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

from sciosci.assets import text_assets as ta
from DEC.DEC_keras import DEC_simple_run
np.random.seed(100)
# =============================================================================
# Evaluation method
# =============================================================================
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

# =============================================================================
# Models
# =============================================================================
classifier = 'deep'

# def baseline_model(model_shape:list=[200,100,50,10],input_dim:int=128,act:str='relu',act_last:str='softmax',loss_f:str='categorical_crossentropy',opt:str='adam'):
def baseline_model(model_shape:list,input_dim:int=128,act:str='relu',act_last:str='softmax',loss_f:str='categorical_crossentropy',opt:str='adam'):
    def model():
        nn = Sequential()
        nn.add(Dense(model_shape[0], input_dim=input_dim, activation=act, name='Dense_1'))#, kernel_initializer='he_uniform'))
        for i,dim in enumerate(model_shape[1:-1]):
            nn.add(Dense(dim, activation=act, name="Dense_{}".format(i+2)))
        nn.add(Dense(model_shape[-1], activation=act_last, name="Dense_{}".format(i+3)))
        # compile model
        # opt = SGD(lr=0.01, momentum=0.9)
        # opt = Adam(learning_rate=0.1)
        nn.compile(loss=loss_f, optimizer=opt, metrics=['accuracy'])  #sparse_categorical_crossentropy  if you don't want to categorize yourself
        print(nn.summary())
        return nn
    return model()

def baseline_model2():
    # Create model here
    model = Sequential()
    model.add(Dense(15, input_dim = 128, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dense(15, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def baseline_model3():
    # Create model here
    model = Sequential()
    model.add(Dense(400, input_dim = input_dim, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dense(200, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score ,average = 'micro'),
           'recall' : make_scorer(recall_score ,average = 'micro'), 
           'f1_score' : make_scorer(f1_score ,average = 'micro')}


# =============================================================================
# Run
# =============================================================================
method_a = False    # Method A: Using sklearn kfold
method_b = True     # Method B: Using custom kfold loop
# datapath = '/mnt/16A4A9BCA4A99EAD/GoogleDrive/Data/'
# datapath = '/mnt/6016589416586D52/Users/z5204044/GoogleDrive/GoogleDrive/Data/'
datapath = '/home/sahand/GoogleDrive/Data/'
data_dir =  datapath+"Corpus/cora-classify/cora/"
label_address =  data_dir+"clean/single_component_small_19k/labels"
n_folds = 5

# vec_file_names = ['embeddings/node2vec super-d2v-node 128-80-10 p4q1','embeddings/node2vec super-d2v-node 128-80-10 p1q025','embeddings/node2vec super-d2v-node 128-10-100 p1q025']#,'Doc2Vec patent corpus',
                  # ,'embeddings/node2vec-80-10-128 p1q0.5','embeddings/node2vec deepwalk 80-10-128']
# vec_file_names =  ['embeddings/node2vec super-d2v-node 128-80-10 p1q05']
# vec_file_names =  ['embeddings/single_component_small_18k/doc2vec 300D dm=1 window=10']
# vec_file_names =  [
#                     # 'embeddings/single_component_small/deep_nonlinear_embedding_600',
#                     'embeddings/single_component_small_18k/cocite dw 300-70-20 p1q1',
#                     'embeddings/single_component_small_18k/cocite node2vec 300-70-20 p1q05',
#                     'embeddings/single_component_small_18k/cocite sn dw 300-70-20 p1q1',
#                     'embeddings/single_component_small_18k/cocite sn node2vec 300-70-20 p1q05',
#                     'embeddings/single_component_small_18k/cocite TADW-120-240-bow',
#                     'embeddings/single_component_small_18k/cocite TADW-120-240-tfidf',
#                     'embeddings/single_component_small_18k/cocite TENE-120-240-bow',
#                     'embeddings/single_component_small_18k/cocite TENE-120-240-tfidf',
#                     'embeddings/single_component_small_18k/dw 300-70-20 p1q1',
#                     'embeddings/single_component_small_18k/n2v 300-70-20 p1q05'
#                     ]
vec_file_names =  [
                   'embeddings/single_component_small_19k/doc2vec 300D dm=1 window=10',
                   
                   'embeddings/single_component_small_19k/node2vec 300-70-20 p1q05',
                   'embeddings/single_component_small_19k/node2vec 300-70-20 p1q05 with_d2v300D_supernodes',
                   
                   'embeddings/single_component_small_19k/DW 300-70-20',
                   'embeddings/single_component_small_19k/DW 300-70-20 with_d2v300D_supernodes',
                   
                   'embeddings/single_component_small_19k/naive d2v + dw 600',
                   'embeddings/single_component_small_19k/naive d2v + n2v 600',
                   
                   'embeddings/single_component_small_19k/TADW-120-240-tfidf',
                   'embeddings/single_component_small_19k/TENE-150-300-tfidf'
                   ]

# vec_file_names =  [
    # 'embeddings/with citations/LSTM-merge',
    # 'embeddings/with citations/nonlinear_embedding_200_30percent 2022 oct b',
    # 'embeddings/with citations/nonlinear_embedding_200_30percent 2022 oct c',
    # 'embeddings/with citations/nonlinear_embedding_200_30percent 2022 oct d',
                   # ]
# vec_file_names = ['embeddings/with citations/n2v 100D p2 q1 len5-50']
# filter_file_names = ['clean/single_component_small_18k/nonlinear_test_data_idx 20 april',
#                       'clean/single_component_small_18k/nonlinear_2layers_test_data_idx']
filter_file_names = [
                     # 'clean/with citations/LSTM-merge',
                     # 'clean/with citations/nonlinear_test_data_idx 2022 oct b',
                     # 'clean/with citations/nonlinear_test_data_idx 2022 oct c',
                     # 'clean/with citations/nonlinear_test_data_idx 2022 oct d',

                      False,
                      False,
                      False,
                      False,
                      False,
                      False,
                      False,
                      False,
                      False,
                      False,
                     ]

model_shapes = [
    # [600,300,100,10],
    # # [1024,256,16,10],
    # # [512,64,10],
    [200,100,10,False],
    # [300,200,100,10],
    [400,200,10,True],
    [400,200,10,False]
    ] # last dim is dropout true/false

results_all = []
results_all_detailed = []
for i,file_name in enumerate(vec_file_names):
    print('\nVec file:',file_name)
    file_results = []
    file_results_detailed = []
    gc.collect()
    data_address = data_dir+file_name
    output_dir = data_dir
    # results.append(run_all_tests(data_address,data_dir,labels,model_shapes))

# =============================================================================
# Classify and evaluate
# =============================================================================
# def run_all_tests(data_address:str,output_dir:str,labels:list,model_shapes:list,test_size:float=0.1):
    tic = time.time()
    vectors = pd.read_csv(data_address)#,header=None,)
    
    labels = pd.read_csv(label_address)#,names=['label'])
    labels.columns = ['label']
    
    if filter_file_names[i] is not False:
        data_filter =  datapath+"Corpus/cora-classify/cora/"+filter_file_names[i]
        data_filter = pd.read_csv(data_filter)
        data_filter= data_filter['id_seq'].astype(int).values    
        # vectors = vectors.iloc[data_filter]
        labels = labels.iloc[data_filter]
    
    try:
        vectors = vectors.drop('Unnamed: 0',axis=1)
        print('\nDroped index column. Now '+data_address+' has the shape of: ',vectors.shape)
    except:
        print('\nNo index column detected:',vectors.shape)
        
    try:
        vectors = vectors.drop('id',axis=1)
        print('\nDroped id column. Now '+data_address+' has the shape of: ',vectors.shape)
    except:
        print('\nVector shapes seem to be good:',vectors.shape)
        
    assert labels.shape[0]==vectors.shape[0], "labels dimension mismatch with vec size"
    
    path_to_model = output_dir+'classification/single_component_small_19k/deep/'+data_address.split('/')[-1]
    Path(path_to_model).mkdir(parents=True, exist_ok=True)
    
    # preprocess inputs and split
    # labels_f = pd.factorize(labels.label)[0]
    # enc = OneHotEncoder(handle_unknown='ignore')
    # Y = enc.fit_transform(np.expand_dims(labels_f,axis=1)).toarray()
    Y = pd.get_dummies(labels).values
    X = vectors.values
    
    encoder = LabelEncoder()
    encoder.fit(labels.values.T[0])
    encoded_Y = to_categorical(encoder.transform(labels.values.T[0]))

    factorized_Y = pd.factorize(labels.label)[0]
    
    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=100,shuffle=True)
    # Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=100,shuffle=True)
    kfold = KFold(n_splits = n_folds, shuffle = True, random_state = 100)
    
    callback = EarlyStopping(monitor='val_accuracy', patience=30)
# =============================================================================
#     Random forest
# =============================================================================
    if classifier == 'rf':
        n_trees = 200
        y_single = np.argmax(Y, axis=1)
        model=RandomForestClassifier(n_estimators=n_trees, criterion = 'entropy', random_state = 100,n_jobs=-1) 
        if method_a is True:
            results = cross_validate(model, X, y_single, cv = kfold,scoring=scoring)
        if method_b is True:
            model_results = []
            model_result_detailed = []
            cv_acc = []
            cv_f1 = []
            cv_f1_w = []
            cv_p = []
            cv_p_w = []
            cv_r = []
            cv_r_w = []
            
            fold_number = 0
            for train_index, val_index in kfold.split(X):
                fold_number +=1
                model.fit(X[train_index], y_single[train_index])
                pred = model.predict(X[val_index])
                
                fold_acc = accuracy_score(y_single[val_index], pred)
                fold_f1 = f1_score(y_single[val_index], pred,average='micro')
                fold_f1_weighted = f1_score(y_single[val_index], pred,average='weighted')
                fold_precision = precision_score(y_single[val_index], pred,average='micro')
                fold_precision_weighted = precision_score(y_single[val_index], pred,average='weighted')
                fold_recall = recall_score(y_single[val_index], pred,average='micro')
                fold_recall_weighted = recall_score(y_single[val_index], pred,average='weighted')
                
                cv_acc.append(fold_acc)
                cv_f1.append(fold_f1)
                cv_f1_w.append(fold_f1_weighted)
                cv_p.append(fold_precision)
                cv_p_w.append(fold_precision_weighted)
                cv_r.append(fold_recall)
                cv_r_w.append(fold_recall_weighted)
                
                print('writing to disk...')
                y_true = y_single[val_index]
                y_pred = pred
                pd.DataFrame(y_true).to_csv(path_to_model+"/y_true k5 fold"+str(fold_number),index=False)
                pd.DataFrame(y_pred).to_csv(path_to_model+"/y_pred k5 fold"+str(fold_number),index=False)
                print('done')

            model_results.append(np.mean(cv_acc))
            model_result_detailed.append({'accuracy':cv_acc,
                                    'f1':cv_f1,
                                    'f1 weighted':cv_f1_w,
                                    'precision':cv_p,
                                    'precision weighted':cv_p_w,
                                    'recall':cv_r,
                                    'recall weighted':cv_r_w})
            print('\n===\nResults for random forest:')
            print('accuracy',cv_acc,
                                    'f1',cv_f1,
                                    'f1 weighted',cv_f1_w,
                                    'precision',cv_p,
                                    'precision weighted',cv_p_w,
                                    'recall',cv_r,
                                    'recall weighted',cv_r_w,
                                    '\n')

            file_results.append({'n_trees':str(n_trees),'results':model_results})
            file_results_detailed.append({'n_trees':str(n_trees),'results':model_result_detailed})

    toc = time.time()
    print('All done in '+str(toc - tic)+'seconds!')
    results_all.append({'file':file_name,'results':file_results})
    results_all_detailed.append({'file':file_name,'results':file_results_detailed})
# =============================================================================
#     DNN
# =============================================================================
    if classifier == 'deep':

        for i,model_shape in enumerate(model_shapes):
            model_results = []
            model_result_detailed = []
            #run models
            # print(model_shape)
            # try:
            #     del model_tmp
            # except:
            #     pass
            # model_tmp = baseline_model(model_shape)
            # dot_img_file = 'models/classifiers/'+str(model_shape)+'.png'
            # tf.keras.utils.plot_model(model_tmp, to_file=dot_img_file, show_shapes=True)
            # estimator = KerasClassifier(build_fn = model, epochs = 100, batch_size = 10, verbose = 0)
            
            # Method A: Using sklearn kfold
            if method_a is True:
                model = KerasClassifier(build_fn = baseline_model(model_shape=model_shape,input_dim=X.shape[1]), epochs = 100, batch_size = 32, verbose = 1, callbacks=[callback])
                result = cross_val_score(model, X, Y, cv = kfold)
                # result = cross_val_score(model, X, Y, cv = kfold)
                print(" >> Baseline accuracy:",result.mean()*100," ~",result.std()*100)
                model_result_detailed.append(result)
                model_results.append(result.mean()*100)
                
            # Method B: Using custom kfold
            if method_b is True:
                cv_acc = []
                cv_f1 = []
                cv_f1_w = []
                cv_p = []
                cv_p_w = []
                cv_r = []
                cv_r_w = []
                y_single = np.argmax(Y, axis=1)
                fold_number = 0
                for train_index, val_index in kfold.split(X):
                    try:
                        del model
                    except:
                        pass
                    fold_number +=1
                    input_dim = X.shape[1]
                    model_shape = model_shape
                    # def baseline_model1(act:str='relu',act_last:str='softmax',loss_f:str='categorical_crossentropy',opt:str='adam'):
                    #     nn = Sequential()
                    #     nn.add(Dense(model_shape[0], input_dim=input_dim, activation=act, name='Dense_1'))#, kernel_initializer='he_uniform'))
                    #     if model_shape[-1]:
                    #         nn.add(Dropout(0.2))
                    #     for i,dim in enumerate(model_shape[1:-2]):
                    #         nn.add(Dense(dim, activation=act, name="Dense_{}".format(i+2)))
                    #     nn.add(Dense(model_shape[-2], activation=act_last, name="Dense_{}".format(i+3)))
                    #     # compile model
                    #     # opt = SGD(lr=0.01, momentum=0.9)
                    #     # opt = Adam(learning_rate=0.1)
                    #     nn.compile(loss=loss_f, optimizer=opt, metrics=['accuracy'])  #sparse_categorical_crossentropy  if you don't want to categorize yourself
                    #     print(nn.summary())
                        # return nn
            
                    # model = KerasClassifier(build_fn=baseline_model1, batch_size=64, epochs=100,callbacks=[callback])
                    
                    act='relu'
                    act_last='softmax'
                    loss_f='categorical_crossentropy'
                    opt='adam'
                    model = Sequential()
                    model.add(Dense(model_shape[0], input_dim=input_dim, activation=act, name='Dense_1'))#, kernel_initializer='he_uniform'))
                    if model_shape[-1]:
                        model.add(Dropout(0.2))
                    for i,dim in enumerate(model_shape[1:-2]):
                        model.add(Dense(dim, activation=act, name="Dense_{}".format(i+2)))
                    model.add(Dense(model_shape[-2], activation=act_last, name="Dense_last"))
                    # compile model
                    # opt = SGD(lr=0.01, momentum=0.9)
                    # opt = Adam(learning_rate=0.1)
                    model.compile(loss=loss_f, optimizer=opt, metrics=['accuracy'])  #sparse_categorical_crossentropy  if you don't want to categorize yourself
                    print(model.summary())
                    
                    64# model = KerasClassifier(build_fn=baseline_model3, batch_size=32, epochs=100,callbacks=[callback])
                    model.fit(X[train_index], Y[train_index],validation_data=(X[val_index], Y[val_index]),batch_size=64,callbacks=[callback],epochs=100)
                    
                    # pred = np.argmax(model.predict(X[val_index]))#model.predict_classes(X[val_index])
                    
                    pred = np.argmax(model.predict(X[val_index]), axis=-1)

                    # get fold accuracy & append
                    fold_acc = accuracy_score(y_single[val_index], pred)
                    fold_f1 = f1_score(y_single[val_index], pred,average='micro')
                    fold_f1_weighted = f1_score(y_single[val_index], pred,average='weighted')
                    fold_precision = precision_score(y_single[val_index], pred,average='micro')
                    fold_precision_weighted = precision_score(y_single[val_index], pred,average='weighted')
                    fold_recall = recall_score(y_single[val_index], pred,average='micro')
                    fold_recall_weighted = recall_score(y_single[val_index], pred,average='weighted')
                    file_name_short = file_name.split('/')[-1]
                    model.save('./models/classification/{file_name_short}-{model}-{fold_acc}-{fold_f1}-{fold_f1_weighted}.h5'.format(file_name_short=file_name_short,model=str(model_shape),fold_acc=str(round(fold_acc, 4)),fold_f1=str(round(fold_f1, 4)),fold_f1_weighted=str(round(fold_f1_weighted, 4))))
                    
                    
                    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_last').output)
                    embeddings = pd.DataFrame(intermediate_layer_model.predict(X))
                    embeddings.to_csv('./models/classification/vecs/{file_name_short}-{model}-{fold_acc}-{fold_f1}-{fold_f1_weighted}--vecs.csv'.format(file_name_short=file_name_short,model=str(model_shape),fold_acc=str(round(fold_acc, 4)),fold_f1=str(round(fold_f1, 4)),fold_f1_weighted=str(round(fold_f1_weighted, 4))),index=False)
                    
                    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_2').output)
                    embeddings = pd.DataFrame(intermediate_layer_model.predict(X))
                    embeddings.to_csv('./models/classification/vecs_b/{file_name_short}-{model}-{fold_acc}-{fold_f1}-{fold_f1_weighted}--vecs.csv'.format(file_name_short=file_name_short,model=str(model_shape),fold_acc=str(round(fold_acc, 4)),fold_f1=str(round(fold_f1, 4)),fold_f1_weighted=str(round(fold_f1_weighted, 4))),index=False)
                    
                    
                    cv_acc.append(fold_acc)
                    cv_f1.append(fold_f1)
                    cv_f1_w.append(fold_f1_weighted)
                    cv_p.append(fold_precision)
                    cv_p_w.append(fold_precision_weighted)
                    cv_r.append(fold_recall)
                    cv_r_w.append(fold_recall_weighted)

                    print('writing to disk...')
                    y_true = y_single[val_index]
                    y_pred = pred
                    pd.DataFrame(y_true).to_csv(path_to_model+"/y_true k5 fold"+str(fold_number),index=False)
                    pd.DataFrame(y_pred).to_csv(path_to_model+"/y_pred k5 fold"+str(fold_number),index=False)
                    print('done')

                model_results.append(np.mean(cv_acc))
                model_result_detailed.append({'accuracy':cv_acc,
                                        'f1':cv_f1,
                                        'f1 weighted':cv_f1_w,
                                        'precision':cv_p,
                                        'precision weighted':cv_p_w,
                                        'recall':cv_r,
                                        'recall weighted':cv_r_w})
                print('\n===\nResults for',model_shape,':')
                print('accuracy',cv_acc,
                                        'f1',cv_f1,
                                        'f1 weighted',cv_f1_w,
                                        'precision',cv_p,
                                        'precision weighted',cv_p_w,
                                        'recall',cv_r,
                                        'recall weighted',cv_r_w,
                                        '\n')

            file_results.append({'model':str(model_shape),'results':model_results})
            file_results_detailed.append({'model':str(model_shape),'results':model_result_detailed})
            
            
            # fit model
            # path_to_model = path_to_model + '/' + str(model_shape) + '_checkpoint.hdf5'
            # print('The model will be saved into ',path_to_model)
            # checkpoint = ModelCheckpoint(filepath=path_to_model, save_best_only=True, monitor='val_accuracy', mode='max')
            
            # history = model.fit(Xtrain, Ytrain, validation_data=(Xvalid, Yvalid), epochs=100, verbose=0, callbacks=[checkpoint])
            
            # #test model
            # # loss, accuracy, f1_score, precision, recall
            # Ypred = model.predict(Xtest)
            # results.append(model.evaluate(Xtest, Ytest, verbose=0)+[f1_m(Ytest,Ypred),precision_m(Ytest,Ypred),recall_m(Ytest, Ypred)])
            
            # _, train_acc = model.evaluate(Xtrain, Ytrain, verbose=0)
            # _, test_acc = model.evaluate(Xtest, Ytest, verbose=0)
            
            # print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
            
            # # plot loss during training
            # pyplot.subplot(211)
            # pyplot.title('Loss')
            # pyplot.plot(history.history['loss'], label='train')
            # pyplot.plot(history.history['val_loss'], label='test')
            # pyplot.legend()
            # # plot accuracy during training
            # pyplot.subplot(212)
            # pyplot.title('Accuracy')
            # pyplot.plot(history.history['accuracy'], label='train')
            # pyplot.plot(history.history['val_accuracy'], label='test')
            # pyplot.legend()
            # pyplot.title(str(model_shape))
            # pyplot.show()
        
        # column_names = ['loss','accuracy','f1','percision','recall']
        # results = pd.DataFrame(results,columns=column_names)
        
    toc = time.time()
    print('All done in '+str(toc - tic)+'seconds!')
    results_all.append({'file':file_name,'results':file_results})
    results_all_detailed.append({'file':file_name,'results':file_results_detailed})

filename = output_dir+'classification/single_component_small_19k/'+classifier+'/dict results k'+str(n_folds)+".txt"
if os.path.exists(filename):
    append_write = 'a' # append if already exists
else:
    append_write = 'w' # make a new file if not

text_file = open(filename, append_write)
text_file.write(str(results_all_detailed))
text_file.close()

counter = 0
results_clean = {}
for i,result in tqdm(enumerate(results_all_detailed)):
    for j,model in enumerate(result['results']):
        results_clean[counter] = {'data':result['file']}
        results_clean[counter]['model']= model['model']
        results_clean[counter]['max_pr'] = max([d for d in model['results'][0]['precision']])
        results_clean[counter]['max_pr_w'] = max([d for d in model['results'][0]['precision weighted']])
        results_clean[counter]['max_acc'] = max([d for d in model['results'][0]['accuracy']])
        results_clean[counter]['max_f1'] = max([d for d in model['results'][0]['f1']])
        results_clean[counter]['max_f1_w'] = max([d for d in model['results'][0]['f1 weighted']])
        results_clean[counter]['max_re'] = max([d for d in model['results'][0]['recall']])
        results_clean[counter]['max_re_w'] = max([d for d in model['results'][0]['recall weighted']])
        
        results_clean[counter]['avg_pr'] = st.mean([d for d in model['results'][0]['precision']])
        results_clean[counter]['avg_pr_w'] = st.mean([d for d in model['results'][0]['precision weighted']])
        results_clean[counter]['avg_acc'] = st.mean([d for d in model['results'][0]['accuracy']])
        results_clean[counter]['avg_f1'] = st.mean([d for d in model['results'][0]['f1']])
        results_clean[counter]['avg_re'] = st.mean([d for d in model['results'][0]['recall weighted']])
        results_clean[counter]['avg_re_w'] = st.mean([d for d in model['results'][0]['recall weighted']])
        
        results_clean[counter]['std_pr'] = st.stdev([d for d in model['results'][0]['precision']])
        results_clean[counter]['std_pr_w'] = st.stdev([d for d in model['results'][0]['precision weighted']])
        results_clean[counter]['std_acc'] = st.stdev([d for d in model['results'][0]['accuracy']])
        results_clean[counter]['std_f1'] = st.stdev([d for d in model['results'][0]['f1']])
        results_clean[counter]['std_re'] = st.stdev([d for d in model['results'][0]['recall weighted']])
        results_clean[counter]['std_re_w'] = st.stdev([d for d in model['results'][0]['recall weighted']])
        counter+=1
    
results_all_detailed_df = pd.DataFrame(results_clean).T
results_all_detailed_df_g = results_all_detailed_df.groupby(['data'])['max_acc'].max()
results_all_detailed_df_g = results_all_detailed_df.groupby(['data'])['avg_acc'].max()

filename = output_dir+'classification/single_component_small_19k/'+classifier+'/df results k'+str(n_folds)+".txt"
results_all_detailed_df.to_csv(filename)

# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
# f1_score(y_true, y_pred, average='micro')


# f1_score(y_true, y_pred, average='micro')





