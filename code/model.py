import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as functional
import torch.nn as nn
from preprocess import PreprocessTweets
import pandas as pd

from typing import Union
from torch.utils.data import DataLoader

# class FancyModel():
#     # INSERT MODEL HERE PLZ
#     pass

BATCH_SIZE = 500

class CNN():
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 51, kernel_size=5, stride=1, padding=2),  # 10 x (28 x 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14 x 14
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(51, 102, kernel_size=5, stride=1, padding=2),  # 20 x (14 x 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.layer_3 = nn.Linear(20 * 7 * 7, 64)
        self.relu = nn.ReLU()
        self.layer_4 = nn.Linear(64, 10)
        return

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.flatten(x)
        x = self.layer_3(x)
        x = self.relu(x)
        x = self.layer_4(x)
        return functional.log_softmax(x, dim=1)


net = CNN()
print(net)


def get_train_test() -> Union[DataLoader, DataLoader]:
    pt = PreprocessTweets()
    X_train, X_test, y_train, y_test = pt.process()

    benni = X_train.to_numpy()

    sara = torch.Tensor(benni)
    emil = torch.Tensor(y_train.to_numpy());
    train_data = torch.utils.data.TensorDataset(sara, emil)


    test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()),
                                               torch.Tensor(y_test.to_numpy()))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                             shuffle=False)

    return train_loader, test_loader



training_set, test_set = get_train_test()


# Train the network
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
for epocs in range(2):
  for data in training_set:
    X, y = data
    optimizer.zero_grad()
    output = net(X)
    loss = functional.nll_loss(output, y)
    loss.backward()
    optimizer.step()
  print(loss)

  # Evaluate.
  total = 0
  correct = 0
  with torch.no_grad():
      for data in test_set:
          X, y = data;
          output = net(X)  # Forward pass
          for idx, val in enumerate(output):
              if (torch.argmax(val) == y[idx]):
                  correct += 1
              total += 1
  print("Accuracy: ", round(correct / total, 3))