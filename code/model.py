# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:55:19 2020

@authors: Benedikt, Emil, Sara
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional


class RNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.embedding = nn.Embedding(30522, 16)
        self.LSTM = nn.LSTM(input_size=16,
                            hidden_size=8,
                            num_layers=1,
                            batch_first=True)
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
  
    def forward(self, x):

        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        # x = self.LSTM(x)
        # print(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        # print(x.shape)
        
        return x#functional.log_softmax(x, dim=1)

'''    
from preprocess import PreprocessTweets
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split


prepro = PreprocessTweets()
prepro.load_data()
prepro.clean_data()
prepro.tokenize()
prepro.trimming()
prepro.padding()
prepro.return_numpy()

train_data = prepro.train
test_data = prepro.test

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)


X_train, X_test, y_train, y_test = train_test_split(train_data, test_data)


'''













