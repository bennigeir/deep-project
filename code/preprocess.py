# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:56:31 2020

@authors: Benedikt, Emil, Sara
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils import (remove_url,
                   remove_non_alpha,
                   pad_list,
                   trim_list,
                   remove_stop_words)


class PreprocessTweets():
    
    def __init__(self, max_seq_len):
        
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
        
        self.max_seq_len = max_seq_len
        self.stop_words = set(stopwords.words('english'))
        
    
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
    
    
    def strip_stop_words(self):
        # Remove stopwords?
                
        self.train['OriginalTweet'] = self.train['OriginalTweet'].apply(lambda x: remove_stop_words(x, self.stop_words))
        self.test['OriginalTweet'] = self.test['OriginalTweet'].apply(lambda x: remove_stop_words(x, self.stop_words))
        
    
    def get_target(self, val):
        # Encode target values
        encoder = LabelEncoder()
        
        assert val in [5,3,2], 'val must have values 5, 3, or 2'
        
        if val == 3:
            self.train.loc[self.train['Sentiment'] == 'Extremely Positive'] = 'Positive'
            self.test.loc[self.test['Sentiment'] == 'Extremely Positive'] = 'Positive'
            
            self.train.loc[self.train['Sentiment'] == 'Extremely Negative'] = 'Negative'
            self.test.loc[self.test['Sentiment'] == 'Extremely Negative'] = 'Negative'
            
        if val == 2:
            self.train.loc[self.train['Sentiment'] == 'Extremely Positive'] = 'Positive'
            self.test.loc[self.test['Sentiment'] == 'Extremely Positive'] = 'Positive'
            
            self.train.loc[self.train['Sentiment'] == 'Extremely Negative'] = 'Negative'
            self.test.loc[self.test['Sentiment'] == 'Extremely Negative'] = 'Negative'
            
            self.train = self.train[self.train['Sentiment'] != 'Neutral']
            self.test = self.test[self.test['Sentiment'] != 'Neutral']
        
        self.train['Sentiment'] = encoder.fit_transform(self.train['Sentiment'])
        self.test['Sentiment'] = encoder.fit_transform(self.test['Sentiment'])
    
    
    def return_numpy(self):

        self.train = self.train.to_numpy()
        self.test = self.test.to_numpy()