#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:31:43 2021

@author: github.com/sahandv
"""
import pandas as pd
import torch
torchversion = torch.__version__
import torch.nn.functional as F
import torch.nn as nn
# from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv
# from torch_geometric.utils import to_networkx
# from torch_geometric.utils import degree
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


def mask_maker(data_range:int,train_split:float=0.75,val_split:float=0.2,test_split:float=0.05):
    """
    Parameters
    ----------
    data_range : int
        Dataset size.
    train_split : float, optional
        The default is 0.75.
    val_split : float, optional
        The default is 0.2.
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
    i_train,i_test = train_test_split(indices,test_size=1-train_split)
    i_validation, i_test = train_test_split(i_test,test_size=test_split/test_valid_split)
    indices = pd.DataFrame(indices,columns=['id'])
    train_mask = indices.id.isin(i_train).values
    validation_mask = indices.id.isin(i_validation).values
    test_mask = indices.id.isin(i_test).values
    return train_mask,validation_mask,test_mask
    
# =============================================================================
# GCN
# =============================================================================
class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    # self.gcn12 = GCNConv(dim_h, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_out)
    # self.dense = nn.Linear(100,dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.5, training=self.training)
    h = self.gcn1(h, edge_index)
    h = torch.relu(h)
    # h = self.gcn12(h, edge_index)
    # h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn2(h, edge_index)
    # h = self.dense(h)
    # h = torch.relu(h)
    return h, F.log_softmax(h, dim=1)

# =============================================================================
# GAT
# =============================================================================
class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    # self.gat12 = GATv2Conv(dim_h*heads, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    # h = self.gat12(h, edge_index)
    # h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

# =============================================================================
# 
# =============================================================================
def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()



def train_td(model, data):
    """Train a GNN model and return the trained model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 300

    model.train()
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')
          
    return model

def test_td(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=1)[data.test_mask]
    y = data.y[data.test_mask]
    acc = accuracy(y_pred,y)
    f1 = f1_score(y.cpu().tolist(), y_pred.cpu().tolist(),average='micro')
    precision = precision_score(y.cpu().tolist(), y_pred.cpu().tolist(),average='micro')
    
    return acc,f1,precision

# =============================================================================
# 
# =============================================================================

class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
def __len__(self):
        return len(self.labels)
def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = {"Text": text, "Class": label}
        return sample

































