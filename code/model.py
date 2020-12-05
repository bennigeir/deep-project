# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:55:19 2020

@authors: Benedikt, Emil, Sara
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.embedding = nn.Embedding(30522, 64)
        
        self.LSTM = nn.LSTM(input_size=64,
                            hidden_size=32,
                            batch_first=True,
                            bidirectional=True)
        
        self.drop = nn.Dropout(0.2)

        self.fc = nn.Linear(32, 5)
        
        self.act = nn.Softmax()
        
  
    def forward(self, x):

        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        
        x = self.drop(x)
        x_pack = pack_padded_sequence(x, torch.Tensor(500), batch_first=True)
        
        lstm_out, (ht, ct) = self.LSTM(x)
        # print(x)
        x = self.fc(ht[-1])

        
        return self.act(x)
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













