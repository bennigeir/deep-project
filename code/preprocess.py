# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:56:31 2020

@authors: Benedikt, Emil, Sara
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
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

        self.train = self.test = None
        
        self.max_seq_len = 50
        
    
    def load_data(self):
        
        # Read raw csv files
        train_data = pd.read_csv(self.train_path, encoding = self.encoding)
        test_data = pd.read_csv(self.test_path, encoding = self.encoding)
        
        # Split data into features and targets
        self.train = train_data[self.features]
        self.test = test_data[self.features]

    
    def clean_data(self):
        
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: remove_url(x))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: remove_url(x))
        
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: remove_non_alpha(x))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: remove_non_alpha(x))
        
        # Cast tweets to lower case
        self.train['OriginalTweet'] = self.train['OriginalTweet'].str.lower()
        self.test['OriginalTweet'] = self.test['OriginalTweet'].str.lower()
    
    
    def tokenize(self):
        
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: word_tokenize(x))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: word_tokenize(x))
    
    
    def trimming(self):

        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: trim_list(x, self.max_seq_len))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: trim_list(x, self.max_seq_len))
    
    
    def padding(self):
        
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: pad_list(x, self.max_seq_len))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: pad_list(x, self.max_seq_len))
        
    
    def vocabulary(self):
        # Frequency distribution?
        pass
    
    
    def stop_words(self):
        # Remove stopwords?
        pass
    
    
    def get_target(self):
        # Encode target values
        encoder = LabelEncoder()
        
        self.train.loc[self.train['Sentiment'] == 'Extremely Positive'] = 'Positive'
        self.test.loc[self.test['Sentiment'] == 'Extremely Positive'] = 'Positive'
        
        self.train.loc[self.train['Sentiment'] == 'Extremely Negative'] = 'Negative'
        self.test.loc[self.test['Sentiment'] == 'Extremely Negative'] = 'Negative'
        
        self.train['Sentiment'] = encoder.fit_transform(self.train['Sentiment'])
        self.test['Sentiment'] = encoder.fit_transform(self.test['Sentiment'])
    
    
    def return_numpy(self):

        self.train = self.train[['OriginalTweet','Sentiment']].to_numpy()
        self.test = self.test[['OriginalTweet','Sentiment']].to_numpy()