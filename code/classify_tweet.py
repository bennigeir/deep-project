# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:56:31 2020

@author: Benedikt
"""

import pandas as pd


def read_data():
    train_data = pd.read_csv('../data/Corona_NLP_train.csv', encoding = "ISO-8859-1")#, encoding='utf-8')
    test_data = pd.read_csv('../data/Corona_NLP_test.csv', encoding = "ISO-8859-1")#, encoding='utf-8')
    
    return train_data, test_data


train_data, test_data = read_data()