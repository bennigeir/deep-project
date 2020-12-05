# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:55:19 2020

@authors: Benedikt, Emil, Sara
"""

import torch.nn as nn
import torch.nn.functional as functional


class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.embed = nn.Embedding(51, 1, padding_idx=0)
   
        self.layer_1 = nn.Sequential(
            nn.Conv2d(51, 10, kernel_size=5, stride=1, padding=2),  # 10 x (28 x 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14 x 14
        )
  
        self.layer_2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2),  # 20 x (14 x 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  #  20 x (7 x 7)
        )
        self.flatten = nn.Flatten()
        self.layer_3 = nn.Linear(20 * 7 * 7, 64)
        self.relu = nn.ReLU()
        self.layer_4 = nn.Linear(64,10)

  
    def forward(self, x):
        
        x = self.embed(x)
        
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.flatten(x)
        x = self.layer_3(x)
        x = self.relu(x)
        x = self.layer_4(x)
        
        return functional.log_softmax(x, dim=1)