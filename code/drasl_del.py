import torch
import torch.nn as nn
import torch.nn.functional as functional
import matplotlib.pyplot as plt
import json

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

import json
import torch
import numpy as np
import io
import requests

from keras.preprocessing.sequence import pad_sequences
from transformers import (BertForTokenClassification,
                          BertTokenizer,
                          )

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

GPU = True
SAVE_MODEL = True
BATCH_SIZE = 1000
# EPOCHS = 50
EPOCHS = 20
MAX_SEQ_LEN = 75
LR = 0.005
DROPOUT = 0


class LSTM(nn.Module):
    
    def __init__(self, input_size, embed_size, output_size, dropout=0.1):
        
        super().__init__()

        self.name = "lstm"

        self.input_size = input_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        
        self.LSTM = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.embed_size*2,
                            num_layers = 1,
                            batch_first=True,
                            dropout=self.dropout,
                            )
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.embed_size*2, self.output_size)
        
  
    def forward(self, text):
        
        embedded = self.relu(self.embedding(text))
        
        lstm_out, (ht, ct) = self.LSTM(embedded)
        
        return self.fc(ht[-1])


class CNN(nn.Module):
    
    def __init__(self, input_size, embed_size, output_size, dropout=0.1):
        
        super().__init__()

        self.name = "cnn"

        self.input_size = input_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.embedding = nn.Embedding(self.input_size, self.embed_size)

        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = 100, 
                                kernel_size = (3, self.embed_size))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = 100, 
                                kernel_size = (4, self.embed_size))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = 100, 
                                kernel_size = (5, self.embed_size))
        
        self.fc = nn.Linear(3 * 100, self.output_size)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, text):
                      
        embedded = self.embedding(text)        
        embedded = embedded.unsqueeze(1)
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
            
        return self.fc(cat)


class GRU(nn.Module):

    def __init__(self, input_size, embed_size, output_size, dropout=0.1):
        
        super().__init__()

        self.name = "gru"

        self.input_size = input_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout = dropout

        self.embedding = nn.Embedding(self.input_size, self.embed_size)

        self.GRU = nn.GRU(input_size=self.embed_size,
                           hidden_size=self.embed_size * 2,
                           batch_first=True)

        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.embed_size * 2, self.output_size)
        self.act = nn.Softmax()

    def forward(self, text):
        
        embedded = self.relu(self.embedding(text))

        gru_out, hidden = self.GRU(embedded)

        return self.fc(hidden[-1])

def get_model(input_size, embed_size, output_size, model_type, dropout=DROPOUT):
    
    if model_type.lower() == 'lstm':
        return LSTM(input_size, embed_size, output_size, dropout)
        
    if model_type.lower() == 'cnn':
        return CNN(input_size, embed_size, output_size, dropout)

    if model_type.lower() == 'gru':
        return GRU(input_size, embed_size, output_size, dropout)
    
    else:
        return None

def remove_url(tweet):
        return (re.sub(r'http\S+', '', tweet))
    

def remove_non_alpha(tweet):
    return (re.sub(r'[^\x20-\x7E]', '', tweet))


def pad_list(tokens, max_seq_len):
        while len(tokens) <= max_seq_len:
            tokens.append('0')
        return tokens
    
    
def trim_list(tokens, max_seq_len):
    return tokens[0:max_seq_len]


def remove_stop_words(tweet, stop_words):
    temp = [w for w in tweet if not w in stop_words]
    return ' '.join(temp)
    

def clean(inp):
    # Clean the data like in the data prep
    inp = remove_url(inp)
    inp = remove_non_alpha(inp)
    return inp

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def download_file_from_google_drive(id):
    URL = 'https://drive.google.com/uc?export=download'

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    return io.BytesIO(response.content)

def get_stream(model_name):
    if model_name.lower() == 'gru':
        FILE_ID = '1Fz1v0NNu0UWNWWBZ498kUk4sBPENPT-C'
    if model_name.lower() == 'cnn':
        FILE_ID = '1yY5mAy4P-JtfXhPg5A-Ann2iOHeYcB0L'
    if model_name.lower() == 'lstm':
        FILE_ID = '1WYYXAIhf9IFnO4fFV37jZHZMKjzM1Wuh'
        
    dm = download_file_from_google_drive(FILE_ID)
    return dm


def tweet_analysis(inp, model_name):
    assert model_name.lower() in ['gru', 'lstm', 'cnn']

    # get the target mapping
    f = open(model_name + "-mapping.txt")
    
    mapping = json.loads(f.read())
    print(mapping)

    inp = clean(inp)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)

    # Processing done as a list of one tweet
    vocab_size = tokenizer.vocab_size
    data = tokenizer.batch_encode_plus(
        [inp],
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        return_tensors='pt',
        truncation=True
    )

    # Use data loaders - this works but should be refactored... some typing issues
    temp = torch.Tensor(data['input_ids'].float())
    dataa = torch.utils.data.TensorDataset(temp.type(torch.LongTensor))
    loader = DataLoader(dataa, batch_size=1, shuffle=False)

    # Get model
    stream = get_stream(model_name)
    model = get_model(vocab_size, MAX_SEQ_LEN, 3, model_name.lower())
    model.load_state_dict(torch.load(model_name + '.pt'))
    model.eval()

    # Run the model
    with torch.no_grad():
        for val in loader:
            preds = model(val[0])

    print(preds)

    # Select return index of the most likely category, highest value
    values, indices = preds.max(1)

    # return the correct value from the mapping
    return list(mapping.keys())[list(mapping.values()).index(indices.item() + 1)]
    
    
    
    
    
    
    
    
    
    

from flask import Flask, redirect, url_for, render_template, request, jsonify


app = Flask(__name__)


@app.route('/api/v1/analyse_tweet', methods=['GET'])
def get_analysis():
    tweet = str(request.args.get('tweet', ''))
    model_type = str(request.args.get('model_type', ''))

    # default CNN, TODO refactor...
    if len(model_type) == 0:
        model_type = 'cnn'

    ans = tweet_analysis(tweet, model_type)
    return jsonify({'category': ans})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
	app.run()