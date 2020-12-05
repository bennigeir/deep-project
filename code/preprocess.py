# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:56:31 2020

@authors: Benedikt, Emil, Sara
"""

import pandas as pd
import re

from nltk.tokenize import word_tokenize

'''from utils import (remove_url,
                   remove_non_alpha,
                   )'''


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

        self.X_train = self.X_test = self.y_train = self.y_test = None

        self.max_seq_len = 50

    def load_data(self):
        # Read raw csv files
        train_data = pd.read_csv(self.train_path, encoding=self.encoding)
        test_data = pd.read_csv(self.test_path, encoding=self.encoding)

        # Split data into features and targets
        self.X_train = train_data[self.features]
        self.y_train = train_data[self.target]

        self.X_test = test_data[self.features]
        self.y_test = test_data[self.target]

        return self.X_train, self.X_test, self.y_train, self.y_test

    def remove_url(self, tweet):
        return re.sub(r'http\S+', '', tweet)

    def remove_non_alpha(self, tweet):
        return re.sub(r'[^\x20-\x7E]', '', tweet)

    def clean_data(self):
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: self.remove_url(x))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: self.remove_url(x))

        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: self.remove_non_alpha(x))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: self.remove_non_alpha(x))

        # Cast tweets to lower case
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].str.lower()
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].str.lower()

        return self.X_train, self.X_test, self.y_train, self.y_test

    def pad_list(self, l):
        while len(l) <= self.max_seq_len:
            l.append('0')
        return l

    def trim_list(self, l):
        return l[0:self.max_seq_len]

    def tokenize(self):
        # Use nltk tokenize?
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: word_tokenize(x))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: word_tokenize(x))

        return self.X_train, self.X_test, self.y_train, self.y_test

    def trimming(self):
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: self.trim_list(x))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: self.trim_list(x))

        return self.X_train, self.X_test, self.y_train, self.y_test

    def padding(self):
        self.X_train['OriginalTweet'] = self.X_train['OriginalTweet'].apply(lambda x: self.pad_list(x))
        self.X_test['OriginalTweet'] = self.X_test['OriginalTweet'].apply(lambda x: self.pad_list(x))

        return self.X_train, self.X_test, self.y_train, self.y_test

    def vocabulary(self):
        # Frequency distribution?
        pass

    def process(self):
        self.load_data()
        self.clean_data()
        self.tokenize()
        self.trimming()
        self.padding()
        return self.X_train['OriginalTweet'] , self.X_test['OriginalTweet'] , self.y_train['OriginalTweet'] , self.y_test['OriginalTweet']


# prepro = PreprocessTweets()
# X_train, X_test, y_train, y_test = prepro.load_data()
# X_train, X_test, y_train, y_test = prepro.clean_data()
# X_train, X_test, y_train, y_test = prepro.tokenize()
# X_train, X_test, y_train, y_test = prepro.trimming()
# X_train, X_test, y_train, y_test = prepro.padding()
