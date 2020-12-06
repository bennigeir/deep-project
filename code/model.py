# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:55:19 2020

@authors: Benedikt, Emil, Sara
"""

import torch.nn as nn


class LSTM(nn.Module):
    
    def __init__(self, input_size, embed_size):
        super().__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size
        
        self.embedded = nn.Embedding(30522, 50)
        
        self.LSTM = nn.LSTM(input_size=50,
                            hidden_size=75,
                            # num_layers = 1,
                            batch_first=True,
                            # dropout=0.15,
                            )
        
        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(75, 3)
        self.act = nn.Softmax()
        
  
    def forward(self, x):

        x = self.embedded(x)
        x = self.relu(x)  
        
        lstm_out, (ht, ct) = self.LSTM(x)
        
        x = self.fc(ht[-1])
        
        return x


class CNN(nn.Module):
    pass