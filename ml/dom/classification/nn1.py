#!/usr/bin/env python
# coding: utf-8

# In[110]:


import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# In[111]:
random.seed(5)

# In[112]:


cwd = os.getcwd()

# In[113]:


cwd

# In[114]:


path = '../../../data/amazon.dataset.csv'

# read data and apply one-hot encoding
df = pd.read_csv(path)
df.head()
# sns.countplot(x='Label', data=df)

X = df.iloc[0:, 1:].values
Y = df.iloc[0:, 0].values

# In[115]:


X

# In[116]:


Y

# In[117]:


from sklearn.model_selection import train_test_split

x, x_val, y, y_val = train_test_split(X, Y, test_size=0.33, random_state=42)

# In[118]:


x

# In[119]:


x.shape, y.shape, x_val.shape, y_val.shape

# In[120]:


x.shape, y.shape

# In[120]:


# In[121]:


x_train = x.reshape(-1, x.shape[1]).astype('float32')

# In[122]:


y_train = y

x_val = x_val.reshape(-1, x_val.shape[1]).astype('float32')
y_val = y_val

# In[123]:


x_train.shape

# In[124]:


x_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(y_val)

# In[125]:


from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(x).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# In[126]:


data_set = Data()

# In[127]:


trainloader = DataLoader(dataset=data_set, batch_size=10)

# In[128]:


data_set.x[0:10]

# In[129]:


data_set.y[0:10]

# In[130]:


data_set.x.shape, data_set.y.shape

# In[131]:

class Net(nn.Module):
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size, p=0.0):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden1)
        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        self.linear3 = nn.Linear(n_hidden2, n_hidden2)
        nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity='relu')
        self.linear4 = nn.Linear(n_hidden2, out_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.drop(x)
        x = F.relu(self.linear2(x))
        x = self.drop(x)
        x = F.relu(self.linear3(x))
        x = self.drop(x)
        x = self.linear4(x)
        return x

model = Net(276, 50, 30, 7)
model_drop = Net(276, 50, 30, 7, p=0.2)
model_drop

# In[133]:


model_drop.train()

# In[134]:


optimizer_ofit = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# In[135]:


LOSS = {}
LOSS['training data no dropout'] = []
LOSS['validation data no dropout'] = []
LOSS['training data dropout'] = []
LOSS['validation data dropout'] = []

# In[ ]:


# In[136]:


n_epochs = 200

for epoch in range(n_epochs):
    for x, y in trainloader:
        # make a prediction for both models
        yhat = model(data_set.x)
        yhat_drop = model_drop(data_set.x)
        # calculate the lossf or both models
        loss = criterion(yhat, data_set.y)
        loss_drop = criterion(yhat_drop, data_set.y)

        # store the loss for  both the training and validation  data for both models
        LOSS['training data no dropout'].append(loss.item())
        LOSS['training data dropout'].append(loss_drop.item())
        model_drop.eval()
        model_drop.train()

        # clear gradient
        optimizer_ofit.zero_grad()
        optimizer_drop.zero_grad()
        # Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        loss_drop.backward()
        # the step function on an Optimizer makes an update to its parameters
        optimizer_ofit.step()
        optimizer_drop.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

# In[137]:


yhat[0:10]

# In[138]:


torch.max(yhat.data, 1)

# In[139]:


y

# In[140]:


z = model(x_val)
z_dropout = model_drop(x_val)

# In[140]:


# In[141]:


_, yhat = torch.max(z.data, 1)
yhat[0:20]

# In[142]:


_, yhat_dropout = torch.max(z_dropout.data, 1)
yhat_dropout[0:20]

# In[143]:


y_val[0:20]

# In[144]:


# Making the Confusion Matrix
eval_matrix = (pd.crosstab(y_val, yhat))
print(eval_matrix)

# In[145]:


# Making the Confusion Matrix
eval_matrix_dropout = (pd.crosstab(y_val, yhat_dropout))
print(eval_matrix_dropout)

# In[146]:


(eval_matrix[0][0] + eval_matrix[1][1] + eval_matrix[2][2]) / y_val.shape[0]

# In[147]:


# (eval_matrix_dropout[0][0] + eval_matrix_dropout[1][1] + eval_matrix_dropout[2][2]) / y_val.shape[0]
