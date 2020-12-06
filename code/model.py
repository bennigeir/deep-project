# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:55:19 2020

@authors: Benedikt, Emil, Sara
"""

import torch.nn as nn


class LSTM(nn.Module):
    
    def __init__(self, input_size, embed_size, output_size, dropout=0.2):
        super().__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.embedded = nn.Embedding(self.input_size, self.embed_size)
        
        self.LSTM = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.embed_size*2,
                            # num_layers = 1,
                            batch_first=True,
                            # dropout=self.dropout,
                            )
        
        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.embed_size*2, self.output_size)
        self.act = nn.Softmax()
        
  
    def forward(self, x):

        x = self.embedded(x)
        x = self.relu(x)  
        
        lstm_out, (ht, ct) = self.LSTM(x)
        
        x = self.fc(ht[-1])
        
        return x


class CNN(nn.Module):
    pass