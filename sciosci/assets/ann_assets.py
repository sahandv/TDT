#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:31:43 2021

@author: github.com/sahandv
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def plot_graphs(history, string="accuracy", save:str=None):
    """
    Draw from history
    
    Parameters
    ----------
    history : TensorFlow fit history
    string : string
        Draw objective. Default is accuracy. 
    
    Returns
    -------
    None.
    
    """
    
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    if save != None:
        plt.savefig(save)
    plt.show()
    
def accuracy(y_pred, y):
    """Calculate accuracy."""
    return ((y_pred == y).sum() / len(y)).item()

def test_evaluate(y_pred, y):
    """Evaluate the model on test set and print the accuracy score."""
    acc = accuracy(y_pred,y)
    f1 = f1_score(y.tolist(), y_pred.tolist(),average='micro')
    precision = precision_score(y.tolist(), y_pred.tolist(),average='micro')
    
    return acc,f1,precision

def mask_maker(data_range:int,folds:int=None,train_split:float=0.75,test_split:float=0.05):
    """
    Parameters
    ----------
    data_range : int
        Dataset size.
    train_split : float, optional
        The default is 0.75. The remainder goes to val and test splits.

    test_split : float, optional
        This is automatically calculated based on the validation split. The default is 0.05.

    Returns
    -------
    train_mask : nool np.array
    validation_mask : nool np.array
    test_mask : nool np.array

    """
    indices = list(range(data_range))
    test_valid_split = 1-train_split
    if folds==None:
        i_train,i_test = train_test_split(indices,test_size=1-train_split)
        i_validation, i_test = train_test_split(i_test,test_size=test_split/test_valid_split)
        indices = pd.DataFrame(indices,columns=['id'])
        train_mask = indices.id.isin(i_train).values
        validation_mask = indices.id.isin(i_validation).values
        test_mask = indices.id.isin(i_test).values
        return train_mask,validation_mask,test_mask
    else:
        raise('Not implemented yet. Please set a train and test split instead.')
        # rand_indices = random.shuffle(indices)
        # train_mask = []
        # validation_mask = []
        # test_mask = []
        # section_starts = [int(i*len(rand_indices)/folds) for i in range(folds)]
        # section_ends = [min(int((i+1)*len(rand_indices)/folds),len(rand_indices)-1) for i in range(folds)]
        
        # for fold in tqdm(range(folds),total=folds):
        #     train_mask_t = 

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size=64,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_name = df[y_col['name']].nunique()
        self.n_type = df[y_col['type']].nunique()
    
    def on_epoch_end(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return self.n // self.batch_size

# =============================================================================
# https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
# https://keras.io/examples/generative/vae/
# =============================================================================
class VAE(keras.Model):
    def __init__(self, encoder, decoder, epochs, loss_multiplier, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.epochs = epochs
        self.beta = tf.Variable(1.0, trainable=False, name='beta_weight', dtype=tf.float32)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.loss_multiplier = tf.Variable(loss_multiplier, trainable=False, name='loss_multiplier', dtype=tf.float32)
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            mse = keras.losses.MeanSquaredError()
            reconstruction_loss = tf.reduce_mean(mse(data, reconstruction))
            # reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstruction))
            reconstruction_loss *= self.loss_multiplier
            
            # except :
            #     reconstruction_loss = tf.reduce_mean( tf.reduce_sum(mse(tf.expand_dims(data, axis=2), reconstruction)))
                
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         # keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            #         # keras.losses.binary_crossentropy(data, reconstruction))
            #         keras.losses.binary_crossentropy(tf.expand_dims(data, axis=2), reconstruction))
            # )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = tf.reduce_mean(tf.reduce_mean(kl_loss))
            total_loss = reconstruction_loss + (self.beta * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_mean": self.reconstruction_loss_tracker.result(),
            "reconstruction_loss": reconstruction_loss,
            "kl_loss_mean": self.kl_loss_tracker.result(),
            "kl_loss": kl_loss,
            "beta": self.beta
        }

    def test_step(self, data):
        
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        mse = keras.losses.MeanSquaredError()
        reconstruction_loss = tf.reduce_mean(mse(data, reconstruction))
        # reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstruction))
        reconstruction_loss *= self.loss_multiplier
       
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + (self.beta * kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_mean": self.reconstruction_loss_tracker.result(),
            "reconstruction_loss": reconstruction_loss,
            "kl_loss_mean": self.kl_loss_tracker.result(),
            "kl_loss": kl_loss,
        }




# =============================================================================
# AE
# =============================================================================
class AE(keras.Model):
    def __init__(self, encoder, decoder, epochs, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.epochs = epochs
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.total_loss_tracker_val = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean( name="reconstruction_loss")
        self.reconstruction_loss_tracker_val = keras.metrics.Mean(name="reconstruction_loss")
        self.vaelog = []
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            mse = keras.losses.MeanSquaredError()
            reconstruction_loss = mse(data, reconstruction)
            # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(mse(data, reconstruction)))
            
            total_loss = reconstruction_loss 

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_mean": self.reconstruction_loss_tracker.result(),
            "reconstruction_loss": reconstruction_loss
        }
    
    def test_step(self, data):
        z = self.encoder(data)
        reconstruction = self.decoder(z)
        mse = keras.losses.MeanSquaredError()
        reconstruction_loss = mse(data, reconstruction)
        # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(mse(data, reconstruction)))
        total_loss = reconstruction_loss 
        
        self.total_loss_tracker_val.update_state(total_loss)
        self.reconstruction_loss_tracker_val.update_state(reconstruction_loss)
        
        
        
        return {
            "loss": self.total_loss_tracker_val.result(),
            "reconstruction_loss_mean": self.reconstruction_loss_tracker_val.result(),
            "reconstruction_loss": reconstruction_loss
        }
    
    

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(z_log_var / 2) * epsilon
    
class Sampling_2(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = keras.backend.random_normal(tf.shape(log_var))
        term_1 = keras.backend.exp(log_var / 2)
        return mean + term_1 * epsilon
        # return keras.backend.random_normal(tf.shape(log_var)) * keras.backend.exp(log_var / 2) + mean
    
    
class Sampling_3(layers.Layer):
    """
    This sampling is just to test if the reason for this bad loss is the probabilities
    """
    def call(self, inputs):
        mean, log_var = inputs
        term_1 = tf.random.normal(tf.shape(log_var/3))
        term_2 = tf.math.exp(log_var)
        return mean
    
# class WarmUpCallback(keras.callbacks.Callback):
#     def __init__(self, beta, kappa):
#         self.beta = beta
#         self.kappa = kappa
#     # Behavior on each epoch
#     def on_epoch_end(self, epoch, logs={}):
#         if epoch > 10:
#             if keras.backend.get_value(self.beta) <= 1:
#                 keras.backend.set_value(self.beta, max(keras.backend.get_value(self.beta) + self.kappa,1))

class AnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self,name,total_epochs):
        self.name = name
        self.total_epochs=total_epochs
    
    def on_epoch_end(self,epoch,logs={}):
        R = 10
        M = 5
        delay = 5
        kl_max = 1.0
        if self.name=="normal":
            pass
        elif self.name=="monotonic":
            new_value=epoch/float(self.total_epochs)*R
            if new_value>kl_max:
                new_value=kl_max
            tf.keras.backend.set_value(self.model.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.model.beta)))
        elif self.name=="delayed":
            if epoch<=delay:
                tf.keras.backend.set_value(self.model.beta,0)
            else:
                new_value=(epoch-delay)/float(self.total_epochs)*R
                if new_value>kl_max:
                    new_value=kl_max
                tf.keras.backend.set_value(self.model.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.model.beta)))
        elif self.name=="cyclical":
            T=self.total_epochs
            frac=int(T/M)
            tt=((epoch)%frac)/float(frac)
            new_value=tt
            if new_value>kl_max:
                new_value=kl_max
            tf.keras.backend.set_value(self.model.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.model.beta)))
        else:
            tf.keras.backend.set_value(self.model.beta,0)
        
        logs = logs or {}
        logs['beta'] = tf.keras.backend.get_value(self.model.beta)


# =============================================================================
# https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
# https://towardsdatascience.com/graph-attention-networks-in-python-975736ac5c0c
# https://keras.io/examples/graph/gat_node_classification/
# =============================================================================
class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)


"""
### Implement training logic with custom `train_step`, `test_step`, and `predict_step` methods
Notice, the GAT model operates on the entire graph (namely, `node_states` and
`edges`) in all phases (training, validation and testing). Hence, `node_states` and
`edges` are passed to the constructor of the `keras.Model` and used as attributes.
The difference between the phases are the indices (and labels), which gathers
certain outputs (`tf.gather(outputs, indices)`).
"""


class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        learning_rate,
        momentum,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)
        self.optimizer = keras.optimizers.SGD(learning_rate, momentum=momentum)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        indices, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([self.node_states, self.edges])
            # Compute loss
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Compute gradients
        grads = tape.gradient(loss, self.trainable_weights)
        # Apply gradients (update weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute probabilities
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        # Forward pass
        outputs = self([self.node_states, self.edges])
        # Compute loss
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        # Update metric(s)
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}
