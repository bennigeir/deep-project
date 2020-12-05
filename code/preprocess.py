# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:56:31 2020

@authors: Benedikt, Emil, Sara
"""

import pandas as pd

from nltk.tokenize import word_tokenize
from utils import (remove_url,
                   remove_non_alpha,
                   pad_list,
                   trim_list)


class PreprocessTweets():
    
    def __init__(self):
        
        self.train_path = '../data/Corona_NLP_train.csv'
        self.test_path = '../data/Corona_NLP_test.csv'
        self.encoding = 'ISO-8859-1'
        
        self.features = ['UserName',
                         'ScreenName',
                         'Location',
                         'TweetAt',
                         'OriginalTweet',
                         'Sentiment']
        # self.target = ['Sentiment']
        
        # self.X_train = self.X_test = self.y_train = self.y_test = None
        self.train = self.test = None
        
        self.max_seq_len = 50
        
    
    def load_data(self):
        
        # Read raw csv files
        train_data = pd.read_csv(self.train_path, encoding = self.encoding)
        test_data = pd.read_csv(self.test_path, encoding = self.encoding)
        
        # Split data into features and targets
        '''
        self.X_train = train_data[self.features]
        self.y_train = train_data[self.target]
        
        self.X_test = test_data[self.features]
        self.y_test = test_data[self.target]
        '''
        self.train = train_data[self.features]
        self.test = test_data[self.features]

    
    def clean_data(self):
        '''
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: remove_url(x))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: remove_url(x))
        
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: remove_non_alpha(x))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: remove_non_alpha(x))
        
        # Cast tweets to lower case
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].str.lower()
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].str.lower()
        '''
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: remove_url(x))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: remove_url(x))
        
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: remove_non_alpha(x))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: remove_non_alpha(x))
        
        # Cast tweets to lower case
        self.train['OriginalTweet'] = self.train['OriginalTweet'].str.lower()
        self.test['OriginalTweet'] = self.test['OriginalTweet'].str.lower()
    
    
    def tokenize(self):
        # Use nltk tokenize?
        '''
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: word_tokenize(x))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: word_tokenize(x))
        '''
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: word_tokenize(x))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: word_tokenize(x))
    
    
    def trimming(self):
        '''
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: trim_list(x, self.max_seq_len))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: trim_list(x, self.max_seq_len))
        '''
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: trim_list(x, self.max_seq_len))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: trim_list(x, self.max_seq_len))
    
    
    def padding(self):
        '''
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: pad_list(x, self.max_seq_len))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: pad_list(x, self.max_seq_len))
        '''
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: pad_list(x, self.max_seq_len))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: pad_list(x, self.max_seq_len))
        
    
    def vocabulary(self):
        # Frequency distribution?
        pass
    
    
    def stop_words(self):
        # Remove stopwords?
        pass
    
    
    def return_numpy(self):
        '''
        self.X_train = self.X_train['OriginalTweet'].to_numpy()
        self.y_train = self.y_train.to_numpy()
        self.X_test = self.X_test['OriginalTweet'].to_numpy()
        self.y_test = self.y_test.to_numpy()
        '''
        self.train = self.train[['OriginalTweet','Sentiment']].to_numpy()
        self.test = self.test[['OriginalTweet','Sentiment']].to_numpy()

'''
prepro = PreprocessTweets()
X_train, X_test, y_train, y_test = prepro.load_data()
X_train, X_test, y_train, y_test = prepro.clean_data()
X_train, X_test, y_train, y_test = prepro.tokenize()
X_train, X_test, y_train, y_test = prepro.trimming()
X_train, X_test, y_train, y_test = prepro.padding()
'''