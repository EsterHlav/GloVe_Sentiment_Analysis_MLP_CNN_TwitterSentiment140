# Ester Hlav
# Oct 6, 2019
# models.py 

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

from utils import *

##############################################
### MLP model for avg pooled based tensors ###
##############################################

class NetMLP(nn.Module):
    '''
        PyTorch nn.Module for a Multilayer Perceptron.
    '''
    
    def __init__(self, input_size, layer_sizes, activation=nn.ReLU(), 
                 epochs=10, learning_rate=0.001, l2reg=1e-4, dropout=0.1):
        '''
            Constructor for neural network; defines all layers and sets attributes for optimization.  
        '''
        super(NetMLP, self).__init__()
        
        # NN layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_sizes[0])])
        self.layers.extend(nn.Linear(layer_sizes[i-1], layer_sizes[i]) for i in range(1, len(layer_sizes)))
        self.layers.append(nn.Linear(layer_sizes[-1], 1))
         
        # activation functions
        self.activation = activation
        self.finalActivation = nn.Sigmoid()
        
        # optimization attributes
        self.epochs = epochs                
        self.learning_rate = learning_rate
        self.l2reg = l2reg
                           
        # loss and optimizer
        self.criterion = nn.BCELoss()  
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, 
                                         weight_decay=self.l2reg)
                           
    def forward(self, x):
        '''
            Forward pass through MLP.
        '''
        out = x
        # loop through all layers except for last one 
        for layer in self.layers[:-1]:
            out = self.activation(layer(self.dropout(out)))                  
        out = self.finalActivation(self.layers[-1](out))
        return out
     
        
    def compute_loss(self, x, y):
        '''
            Computing loss and evaluation metrics for predictions.


            inputs:
                - x (torch.tensor):      input tensor for neural network
                - y (torch.tensor):      label tensor 

            return:
                - loss (torch.float):    binary cross-entropy loss (BSE) between MLP(x) and y
                - accuracy (float):      accuracy of predictions (sklearn) 
                - precision (float):     precision of predictions (sklearn) 
                - recall (float):        recall of predictions (sklearn) 
                - f1 (float):            F1-score of predictions (sklearn) 
        '''
        # loss
        predictions = self.forward(x)
        loss = self.criterion(predictions, y)
        # binarize predictions from predictions (outputs = 1 if p>0.5 else 0)
        outputs = (predictions>0.5).float()
        # metrics with accuracy, precision, recall, f1
        accuracy, precision, recall, f1 = [metric(y.cpu(), outputs.cpu()) for metric in [accuracy_score, precision_score, recall_score, f1_score]]
        return loss, accuracy, precision, recall, f1
    
    
    def evaluate_loader(self, loader):
        '''
            Computing loss and evaluation metrics for a specific torch.loader.


            inputs:
                - loader (torch.loader):    dataset in torch.loader format

            return:
                - metrics (dict):           mapping of metric name (str) to metric value (float)
        '''
        # compute loss and accuracy for that loader
        metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, }
        # loop over examples of loader
        for i, (x, y) in enumerate(loader): 
            loss, accuracy, precision, recall, f1 = self.compute_loss(x, y)
            # sum up metrics in dict
            metrics['loss'] += loss.item()
            metrics['accuracy'] += accuracy
            metrics['precision'] += precision
            metrics['recall'] += recall
            metrics['f1'] += f1
        # normalize all values
        for k in metrics.keys():
            metrics[k]/=len(loader)
        return metrics
                           
        
    def fit(self, train_loader, val_loader, freq_prints=5):
        '''
            Fit a classifier with train and val loaders.


            inputs:
                - train_loader (torch.loader):     training set in torch.loader format
                - val_loader (torch.loader):       validation set in torch.loader format
                - freq_prints (int):               frequency of printing performances of training

            return:
                - history (dict):                  metrics values (metric name to values)
        '''
        # loss, accuracy, precision, recall, f1 init
        history = {'loss': [], 'val_loss': [],
                  'accuracy': [], 'precision': [], 'val_accuracy': [], 'val_precision': [],
                  'recall': [], 'f1': [], 'val_recall': [], 'val_f1': []}
        for epoch in range(self.epochs):
            # one epoch
            train_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, }
            for i, (x, y) in enumerate(train_loader):
                # Forward + Backward + Optimize
                self.optimizer.zero_grad()  # zero the gradient buffer
                loss, accuracy, precision, recall, f1 = self.compute_loss(x, y)
                train_metrics['loss'] += loss.item()
                train_metrics['accuracy'] += accuracy
                train_metrics['precision'] += precision
                train_metrics['recall'] += recall
                train_metrics['f1'] += f1
                # backprop
                loss.backward()
                self.optimizer.step()
            # normalize
            for k in train_metrics.keys():
                train_metrics[k]/=len(train_loader)
                              
            # compute perf on validation set
            val_metrics = self.evaluate_loader(val_loader)
            
            # save metrics in history
            for key in train_metrics:
                history[key].append(train_metrics[key])
            for key in val_metrics:
                history['val_'+key].append(val_metrics[key])
                           
            # printing of performance at freq_prints frequency
            if epoch % freq_prints == 0:
                print ("Epoch {}/{}\nTrain performance: loss={:.3f}, accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}\nVal   performance: loss={:.3f}, accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}".format(
                    epoch+1, self.epochs, history['loss'][-1], history['accuracy'][-1], history['precision'][-1], history['recall'][-1], history['f1'][-1],
                    history['val_loss'][-1], history['val_accuracy'][-1], history['val_precision'][-1], history['val_recall'][-1], history['val_f1'][-1]
                ))
        
        return history
    
    
####################################
### CNN model sequential tensors ###
####################################

class NetCNN(nn.Module):
    '''
        PyTorch nn.Module for a textual CNN.
    '''
    
    def __init__(self, vocab_size, embedding_matrix, filter_sizes=[1,2,3,4,5], num_filters=16, embed_size=200, finetune_emb=False, epochs=10, learning_rate=0.001, l2reg=1e-4, dropout=0.1):
        '''
            Constructor for neural network; defines all layers and sets attributes for optimization.  
        '''
        
        super(NetCNN, self).__init__()
        
        # NN attributes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embed_size = embed_size
        
        # NN layers
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = finetune_emb
        self.convs1 = nn.ModuleList([nn.Conv2d(1, self.num_filters, (K, self.embed_size)) for K in self.filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(self.filter_sizes)*self.num_filters, 1)
        self.finalActivation = nn.Sigmoid()
        
        # optimization attributes
        self.epochs = epochs                
        self.learning_rate = learning_rate
        self.l2reg= l2reg
        
        # loss and optimizer
        self.criterion = nn.BCELoss()  
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, 
                                         weight_decay=self.l2reg)  
    def forward(self, x):
        '''
            Forward pass through CNN.
        '''
        x = self.embedding(x)  
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        out = self.finalActivation(self.fc(x))  
        return out
    
    def compute_loss(self, x, y):
        '''
            Computing loss and evaluation metrics for predictions.


            inputs:
                - x (torch.tensor):      input tensor for neural network
                - y (torch.tensor):      label tensor 

            return:
                - loss (torch.float):    binary cross-entropy loss (BSE) between MLP(x) and y
                - accuracy (float):      accuracy of predictions (sklearn) 
                - precision (float):     precision of predictions (sklearn) 
                - recall (float):        recall of predictions (sklearn) 
                - f1 (float):            F1-score of predictions (sklearn) 
        '''
        # loss
        predictions = self.forward(x)
        loss = self.criterion(predictions, y)
        # binarize predictions from predictions (outputs = 1 if p>0.5 else 0)
        outputs = (predictions>0.5).float()
        accuracy, precision, recall, f1 = [metric(y.cpu(), outputs.cpu()) for metric in [accuracy_score, precision_score, recall_score, f1_score]]
        return loss, accuracy, precision, recall, f1
    
    def evaluate_loader(self, loader):
        '''
            Computing loss and evaluation metrics for a specific torch.loader.


            inputs:
                - loader (torch.loader):    dataset in torch.loader format

            return:
                - metrics (dict):           mapping of metric name (str) to metric value (float)
        '''
        # compute loss and accuracy for that loader
        metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, }
        # loop over examples of loader
        for i, (x, y) in enumerate(loader): 
            loss, accuracy, precision, recall, f1 = self.compute_loss(x, y)
            # sum up metrics in dict
            metrics['loss'] += loss.item()
            metrics['accuracy'] += accuracy
            metrics['precision'] += precision
            metrics['recall'] += recall
            metrics['f1'] += f1
        # normalize all values
        for k in metrics.keys():
            metrics[k]/=len(loader)
        return metrics
                           
        
    def fit(self, train_loader, val_loader, freq_prints=5):
        '''
            Fit a classifier with train and val loaders.


            inputs:
                - train_loader (torch.loader):     training set in torch.loader format
                - val_loader (torch.loader):       validation set in torch.loader format
                - freq_prints (int):               frequency of printing performances of training

            return:
                - history (dict):                  metrics values (metric name to values)
        '''
        # loss, accuracy, precision, recall, f1 init
        history = {'loss': [], 'val_loss': [],
                  'accuracy': [], 'precision': [], 'val_accuracy': [], 'val_precision': [],
                  'recall': [], 'f1': [], 'val_recall': [], 'val_f1': []}
        for epoch in range(self.epochs):
            # one epoch
            train_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, }
            for i, (x, y) in enumerate(train_loader):
                # Forward + Backward + Optimize
                self.optimizer.zero_grad()  # zero the gradient buffer
                loss, accuracy, precision, recall, f1 = self.compute_loss(x, y)
                train_metrics['loss'] += loss.item()
                train_metrics['accuracy'] += accuracy
                train_metrics['precision'] += precision
                train_metrics['recall'] += recall
                train_metrics['f1'] += f1
                # backprop
                loss.backward()
                self.optimizer.step()
            # normalize
            for k in train_metrics.keys():
                train_metrics[k]/=len(train_loader)
                              
            # compute perf on validation set
            val_metrics = self.evaluate_loader(val_loader)
            
            # save metrics in history
            for key in train_metrics:
                history[key].append(train_metrics[key])
            for key in val_metrics:
                history['val_'+key].append(val_metrics[key])
                           
            # printing of performance at freq_prints frequency
            if epoch % freq_prints == 0:
                print ("Epoch {}/{}\nTrain performance: loss={:.3f}, accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}\nVal   performance: loss={:.3f}, accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}".format(
                    epoch+1, self.epochs, history['loss'][-1], history['accuracy'][-1], history['precision'][-1], history['recall'][-1], history['f1'][-1],
                    history['val_loss'][-1], history['val_accuracy'][-1], history['val_precision'][-1], history['val_recall'][-1], history['val_f1'][-1]
                ))
        
        return history
