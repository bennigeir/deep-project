# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:56:31 2020

@author: Benedikt

https://arxiv.org/pdf/1408.5882.pdf


1.  preprocess data
    - OOV, out of vocabulary 
    - variable length sequences, Pytorch: Packed Padding sequence
2.  model
3.  train
4.  evaluate

"""

import pandas as pd


class Preprocess():
    
    def __init__(self):
        
        self.train_path = '../data/Corona_NLP_train.csv'
        self.test_path = '../data/Corona_NLP_test.csv'
        self.encoding = 'ISO-8859-1'
        
        self.features = ['UserName',
                         'ScreenName',
                         'Location',
                         'TweetAt',
                         'OriginalTweet']
        self.target = 'Sentiment'
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    
    def load_data(self):
        
        # Read raw csv files
        train_data = pd.read_csv(self.train_path, encoding = self.encoding)
        test_data = pd.read_csv(self.test_path, encoding = self.encoding)
        
        # Split data into features and targets
        self.X_train = train_data[self.features]
        self.X_test = test_data[self.target]
        
        self.y_train = train_data[self.features]
        self.y_test = test_data[self.target]