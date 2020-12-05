# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:13:48 2020

@author: Benedikt
"""
import torch

from preprocess import PreprocessTweets
from model import CNN
from torch.utils.data import DataLoader


BATCH_SIZE = 500


def get_data():
    
    prepro = PreprocessTweets()
    prepro.load_data()
    prepro.clean_data()
    prepro.tokenize()
    prepro.trimming()
    prepro.padding()
    prepro.return_numpy()

    return prepro.train, prepro.test


def get_model():
    
    cnn_model = CNN()
    pass


train_data, test_data = get_data()


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                          shuffle=False)