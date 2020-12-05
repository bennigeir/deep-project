# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:13:48 2020

@author: Benedikt
"""
import torch

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from preprocess import PreprocessTweets
from model import RNN
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch.nn.functional as functional


BATCH_SIZE = 1000
# EPOCHS = 100
EPOCHS = 16
# BATCH_SIZE = 2000

# %%


def get_data():
    
    prepro = PreprocessTweets()
    prepro.load_data()
    prepro.clean_data()
    # prepro.tokenize()
    # prepro.trimming()
    # prepro.padding()
    prepro.get_target()
    # prepro.return_numpy()

    return prepro.train, prepro.test


def get_model():
    
    cnn_model = RNN()
    pass


train_data, test_data = get_data()


X_train = train_data['OriginalTweet']
y_train = train_data['Sentiment']

X_test = test_data['OriginalTweet']
y_test = test_data['Sentiment']


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)


vocab_size = tokenizer.vocab_size


encoded_data_train = tokenizer.batch_encode_plus(
    X_train, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=50, 
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    X_test, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=50, 
    return_tensors='pt'
)

# %%

model = RNN()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(y_train.values)

input_ids_val = encoded_data_test['input_ids']
attention_masks_val = encoded_data_test['attention_mask']
labels_val = torch.tensor(y_test.values)

dataset_train = TensorDataset(input_ids_train.type(torch.LongTensor), labels_train.type(torch.LongTensor))
dataset_val = TensorDataset(input_ids_val.type(torch.LongTensor), labels_val.type(torch.LongTensor))

dataset_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataset_val = DataLoader(dataset_val, batch_size=1)


# %%

def accuracy_pred(y_pred, y):
    y_pred_tag = torch.argmax(y_pred)
    
    correct_results_sum = (y_pred_tag == y).sum().float()
    acc = correct_results_sum/y.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

# %%
train_accuracy, train_loss_accuracy = [],[]
test_accuracy, test_loss_accuracy = [], []

# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

for epocs in range(EPOCHS):
    
    train_epoch_loss = 0.0
    train_epoch_acc = 0
    
    total = 0
    correct = 0
    model.train()
    for data in dataset_train:
        X, y = data
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad() # Clear the gradients, before next batch.
        output = model(X)  # Forward pass
          
        # print(output, y)
        
        loss = functional.cross_entropy(output, y)#.view(-1,1)) # Computing loss.
        loss.backward()  # Back-propagation (computing gradients)
        optimizer.step() # Update the weights (using gradients).
        
        acc = accuracy_pred(output, y.unsqueeze(1))
        
        # print(acc)
            
        train_epoch_loss += loss.item()
        train_epoch_acc += acc.item()
        
    # train_accuracy.append(train_epoch_acc/len(dataset_train))
    train_loss_accuracy.append(train_epoch_loss/len(dataset_train))
    # print(loss)
    
    
    test_epoch_loss = 0.0
    test_epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for data in dataset_val:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            output = model(X)  # Forward pass
            
            loss = functional.cross_entropy(output, y)  
            # print(loss)
            acc = accuracy_pred(output, y)
            
            test_epoch_loss += loss.item()
            test_epoch_acc += acc.item()
            
    test_accuracy.append(test_epoch_acc/len(dataset_val))
    test_loss_accuracy.append(test_epoch_loss/len(dataset_val))
    
    # print("Accuracy: ", round(correct/total, 3))
  
#%%
  
# Evaluate
total = 0
correct = 0

model.eval()

with torch.no_grad():
  for data in dataset_val:
    X, y = data
    X = X.to(device)
    y = y.to(device)
    output = model(X)  # Forward pass
    #print(output[0])
    for idx, val in enumerate(output):
      if (torch.argmax(val) == y[idx]):
        # print(torch.argmax(val))
        correct += 1
      total += 1
print("Accuracy: ", round(correct/total, 3))

# %%

import matplotlib.pyplot as plt


plt.figure(figsize=(24, 12))
    
plt.plot(train_loss_accuracy, label='train loss')
plt.plot(test_loss_accuracy, label='test loss')
plt.xlabel('Number of epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.suptitle('Loss', fontsize=32)
plt.legend()
plt.show()

plt.figure(figsize=(24, 12))

# plt.plot(train_accuracy, label='train accuracy')
plt.plot(test_accuracy, label='test accuracy')
plt.xlabel('Number of epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.suptitle('Accuracy', fontsize=32)
plt.legend()
plt.show()