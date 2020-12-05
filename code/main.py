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


BATCH_SIZE = 1000

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


# Pytorch TensorDataset Instance
# dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
# dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

dataset_train = TensorDataset(input_ids_train.type(torch.LongTensor), labels_train.type(torch.LongTensor))
dataset_val = TensorDataset(input_ids_val.type(torch.LongTensor), labels_val.type(torch.LongTensor))

dataset_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataset_val = DataLoader(dataset_val, batch_size=1)



import torch.nn.functional as functional

# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

for epocs in range(50):
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
    
    
    '''
    for idx, val in enumerate(output):
        if torch.argmax(output) == y[idx]:
            correct += 1
        total += 1
    '''

  print(loss)
  
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