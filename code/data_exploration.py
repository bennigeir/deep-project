# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:34:42 2020

@author: Benedikt
"""
import matplotlib.pyplot as plt

from preprocess import PreprocessTweets


def get_data(max_seq_len, val):
        
    prepro = PreprocessTweets(max_seq_len)
    prepro.load_data()
    prepro.clean_data()
    # prepro.get_target(val)

    return prepro.train, prepro.test


train_data, test_data = get_data(100, 5)

train_data['Sentiment'].value_counts().plot.pie(autopct='%.2f')

temp_df = train_data[['TweetAt','Sentiment']]
temp_df['count'] = 1
temp_df = temp_df.groupby(['TweetAt','Sentiment']).agg(['count'])
temp_df = temp_df.reset_index()
temp_df.index = temp_df['TweetAt']
temp_df.columns = ['TweetAt','Sentiment','count']


plt.figure(figsize=(24, 12))
plt.plot(temp_df[temp_df['Sentiment'] == 'Extremely Negative']['count'], label='Extremely Negative', color='red')
plt.plot(temp_df[temp_df['Sentiment'] == 'Negative']['count'], label='Negative', color='orange')
plt.plot(temp_df[temp_df['Sentiment'] == 'Neutral']['count'], label='Neutral', color='gray')
plt.plot(temp_df[temp_df['Sentiment'] == 'Positive']['count'], label='Positive', color='greenyellow')
plt.plot(temp_df[temp_df['Sentiment'] == 'Extremely Positive']['count'], label='Extremely Positive', color='green')
plt.gca().invert_xaxis()
plt.legend()
