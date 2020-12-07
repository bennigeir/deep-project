# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:55:19 2020

@authors: Benedikt, Emil, Sara
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    
    def __init__(self, input_size, embed_size, output_size, dropout=0.1):
        
        super().__init__()

        self.name = "lstm"

        self.input_size = input_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        
        self.LSTM = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.embed_size*2,
                            num_layers = 1,
                            batch_first=True,
                            dropout=self.dropout,
                            )
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.embed_size*2, self.output_size)
        
  
    def forward(self, text):
        
        embedded = self.relu(self.embedding(text))
        
        lstm_out, (ht, ct) = self.LSTM(embedded)
        
        return self.fc(ht[-1])


class CNN(nn.Module):
    
    def __init__(self, input_size, embed_size, output_size, dropout=0.1):
        
        super().__init__()

        self.name = "cnn"

        self.input_size = input_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.embedding = nn.Embedding(self.input_size, self.embed_size)

        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = 100, 
                                kernel_size = (3, self.embed_size))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = 100, 
                                kernel_size = (4, self.embed_size))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = 100, 
                                kernel_size = (5, self.embed_size))
        
        self.fc = nn.Linear(3 * 100, self.output_size)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, text):
                      
        embedded = self.embedding(text)        
        embedded = embedded.unsqueeze(1)
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
            
        return self.fc(cat)


class GRU(nn.Module):

    def __init__(self, input_size, embed_size, output_size, dropout=0.1):
        
        super().__init__()

        self.name = "gru"

        self.input_size = input_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout = dropout

        self.embedding = nn.Embedding(self.input_size, self.embed_size)

        self.GRU = nn.GRU(input_size=self.embed_size,
                           hidden_size=self.embed_size * 2,
                           batch_first=True)

        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.embed_size * 2, self.output_size)
        self.act = nn.Softmax()

    def forward(self, text):
        
        embedded = self.relu(self.embedding(text))

        gru_out, hidden = self.GRU(embedded)

        return self.fc(hidden[-1])



































