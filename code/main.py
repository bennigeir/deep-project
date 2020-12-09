# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:13:48 2020

@author: Benedikt
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
import matplotlib.pyplot as plt
import json

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from preprocess import PreprocessTweets
from transformers import BertTokenizer
from model import (LSTM,
                   CNN,
                   GRU)


GPU = True
SAVE_MODEL = False
BATCH_SIZE = 1000
# EPOCHS = 50
EPOCHS = 20
MAX_SEQ_LEN = 75
LR = 0.005
DROPOUT = 0



def get_data(max_seq_len, val):
        
    prepro = PreprocessTweets(max_seq_len)
    prepro.load_data()
    prepro.clean_data()
    # prepro.tokenize()
    # prepro.strip_stop_words()
    # prepro.get_target(val)

    return prepro.train, prepro.test, prepro.mapping


def prepare_data(train_data, test_data):
    
    X_train = train_data['OriginalTweet']
    y_train = train_data['Sentiment']

    X_test = test_data['OriginalTweet']
    y_test = test_data['Sentiment']
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    
    vocab_size = tokenizer.vocab_size
    
    encoded_data_train = tokenizer.batch_encode_plus(
        X_train, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=MAX_SEQ_LEN, 
        return_tensors='pt'
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        X_test, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=MAX_SEQ_LEN, 
        return_tensors='pt'
    )
    
    input_ids_train = encoded_data_train['input_ids']
    labels_train = torch.tensor(y_train.values)
    
    input_ids_val = encoded_data_test['input_ids']
    labels_val = torch.tensor(y_test.values)

    dataset_train = TensorDataset(input_ids_train.type(torch.LongTensor),
                                  labels_train.type(torch.LongTensor))
    dataset_val = TensorDataset(input_ids_val.type(torch.LongTensor),
                                labels_val.type(torch.LongTensor))
    
    dataset_train = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                               shuffle=True)
    dataset_val = DataLoader(dataset_val, batch_size=5000)
    
    return dataset_train, dataset_val, vocab_size


def get_model(input_size, embed_size, output_size, model_type, dropout=DROPOUT):
    
    if model_type.lower() == 'lstm':
        return LSTM(input_size, embed_size, output_size, dropout)
        
    if model_type.lower() == 'cnn':
        return CNN(input_size, embed_size, output_size, dropout)

    if model_type.lower() == 'gru':
        return GRU(input_size, embed_size, output_size, dropout)
    
    else:
        return None


def save_model(model, mapping={}):
    torch.save(model.state_dict(), model.name + ".pt")

    # store the mappings in a file - varies on the data processing
    f = open(model.name + "-mapping.txt", "x")
    f.write(json.dumps(mapping, indent=4))
    f.close()


def run_model(model_type, data_type):
    
    assert model_type in ['lstm', 'cnn', 'gru'], 'Model type invalid'
    
    train_data, test_data, mapping = get_data(MAX_SEQ_LEN, data_type)
    dataset_train, dataset_val, vocab_size = prepare_data(train_data, test_data)

    model = get_model(vocab_size, MAX_SEQ_LEN, data_type, model_type, DROPOUT)
       
    if not model:
        raise RuntimeError('No model selected!')

    if GPU:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    train_accuracy_list, train_loss_list = [], []
    test_accuracy_list, test_loss_list = [], []
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    for epocs in range(EPOCHS):
        
        train_acc, train_loss = train_model(model, dataset_train, device, optimizer)
        test_acc, test_loss = test_model(model, dataset_val, device)
        
        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        
        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss)
        
    plot(train_loss_list, test_loss_list,
         train_accuracy_list, test_accuracy_list,
         model_type.upper(), data_type, LR)

    if SAVE_MODEL:
        save_model(model, mapping)
        

from sklearn.metrics import confusion_matrix
import numpy as np
def accuracy_pred(y_pred, y, test=False):
    
    y_pred_tag = torch.argmax(y_pred)
    
    correct_results_sum = (y_pred_tag == y).sum().float()
    acc = correct_results_sum/y.shape[0]
    acc = torch.round(acc * 100)
    
    # print([element.item() for element in y.flatten()])
    y_pred_list = y_pred.tolist()

    
    confusion = confusion_matrix([element.item() for element in y.flatten()], (np.argmax(y_pred_list, axis=1)).tolist())
    if test:
        print(confusion)
    
    return acc


def train_model(model, dataset_train, device, optimizer):
    
    train_epoch_loss = 0.0
    train_epoch_acc = 0

    model.train()
    
    for data in dataset_train:
        X, y = data
        
        if GPU:
            X = X.to(device)
            y = y.to(device)
            
        optimizer.zero_grad()
        output = model(X)
        
        loss = functional.cross_entropy(output, y)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
        
        acc = accuracy_pred(output, y.unsqueeze(1))
            
        train_epoch_loss += loss.item()
        train_epoch_acc += acc.item()
        
    return (train_epoch_acc/len(dataset_train)), (train_epoch_loss/len(dataset_train))
    
    
def test_model(model, dataset_val, device):
    
    test_epoch_loss = 0.0
    test_epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for data in dataset_val:
            X, y = data
            
            if GPU:
                X = X.to(device)
                y = y.to(device)
            output = model(X)
            
            loss = functional.cross_entropy(output, y)  
            acc = accuracy_pred(output, y, test=True)
            
            test_epoch_loss += loss.item()
            test_epoch_acc += acc.item()
            
    return (test_epoch_acc/len(dataset_val)), (test_epoch_loss/len(dataset_val))


def plot(train_loss_accuracy, test_loss_accuracy,
         train_accuracy, test_accuracy, title, data_type, lr):
    
    
    plt.figure(figsize=(24, 12))
    
    plt.plot(train_loss_accuracy, label='train loss')
    plt.plot(test_loss_accuracy, label='test loss')
    
    plt.xlabel('Number of epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    
    plt.suptitle('Model: {}, Type: {}, Learning Rate: {}'.format(title, data_type, lr), fontsize=28)
    plt.title('Loss', fontsize=32)
    
    # plt.savefig('{}_{}_loss.png'.format(title, data_type))
    
    plt.legend()
    plt.show()
    
    
    
    plt.figure(figsize=(24, 12))
    
    # plt.plot(train_accuracy, label='train accuracy')
    plt.plot(test_accuracy, label='test accuracy')
    
    plt.xlabel('Number of epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    
    plt.suptitle('Model: {}, Type: {}, Learning Rate: {}'.format(title, data_type, lr), fontsize=28)
    plt.title('Accuracy', fontsize=32)
    
    # plt.savefig('{}_{}_acc.png'.format(title, data_type))
    
    plt.legend()
    plt.show()


# %%

        
for i in [[2,'cnn'],[3,'gru'],[5,'gru']]:
    run_model(i[1], i[0])



# %%

from utils import (remove_url,
                   remove_non_alpha,
                   pad_list,
                   trim_list,
                   remove_stop_words)


def clean(inp):
    # Clean the data like in the data prep
    inp = remove_url(inp)
    inp = remove_non_alpha(inp)
    return inp


def tweet_analysis(inp, model_name):
    assert model_name.lower() in ['gru', 'lstm', 'cnn']

    # get the target mapping
    print(model_name)
    f = open(model_name + "-mapping.txt")
    mapping = json.loads(f.read())

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
    model = get_model(vocab_size, MAX_SEQ_LEN, 3, model_name.lower())
    print(model)
    print(model_name + '.pt')
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

# tweet = "Uh-oh, no SpaghettiOs. Panic buying at this San Diego grocery store leaves lots of empty shelves. #Covid_19 #coronavirus #C19 https://t.co/JuSw6pgLng"
# tweet = "Covid made me sick"
# ans = tweet_analysis(tweet, 'cnn')
# print(tweet + " -- IS IN CATEGORY: " + ans)