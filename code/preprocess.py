# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:56:31 2020

@authors: Benedikt, Emil, Sara
"""

import pandas as pd


class PreprocessTweets():
    
    def __init__(self):
        
        self.train_path = '../data/Corona_NLP_train.csv'
        self.test_path = '../data/Corona_NLP_test.csv'
        self.encoding = 'ISO-8859-1'
        
        self.features = ['UserName',
                         'ScreenName',
                         'Location',
                         'TweetAt',
                         'OriginalTweet']
        self.target = ['Sentiment']
        
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
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    
    def clean_data(self):
        # lower, or upper, or both?
        # remoce special characters?
        pass
    
    
    def tokenize(self):
        # Use nltk tokenize?
        pass
    
    
    def padding(self):
        # Tweets have different lengths...
        pass
    
    
    def vocabulary(self):
        # Frequency distribution?
        pass