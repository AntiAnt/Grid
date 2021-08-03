#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from fastai.vision.all import *



# In[2]:


dev = "cuda" if torch.cuda.is_available() else "cpu"


# In[3]:


path = untar_data(URLs.MNIST)
zeroes = (path/'training'/'0').ls().sorted()
ones = (path/'training'/'1').ls().sorted()
twos = (path/'training'/'2').ls().sorted()
threes = (path/'training'/'3').ls().sorted()
fours = (path/'training'/'4').ls().sorted()
fives = (path/'training'/'5').ls().sorted()
sixes = (path/'training'/'6').ls().sorted()
sevens = (path/'training'/'7').ls().sorted()
eights = (path/'training'/'8').ls().sorted()
nines = (path/'training'/'9').ls().sorted()


# In[4]:


zeroes_tensor = [tensor(Image.open(o)) for o in zeroes]
ones_tensor = [tensor(Image.open(o)) for o in ones]
twos_tensor = [tensor(Image.open(o)) for o in twos]
threes_tensor = [tensor(Image.open(o)) for o in threes]
fours_tensor = [tensor(Image.open(o)) for o in fours]
fives_tensor = [tensor(Image.open(o)) for o in fives]
sixes_tensor = [tensor(Image.open(o)) for o in sixes]
sevens_tensor = [tensor(Image.open(o)) for o in sevens]
eights_tensor = [tensor(Image.open(o)) for o in eights]
nines_tensor = [tensor(Image.open(o)) for o in nines]


# In[5]:


tensors = [zeroes_tensor, ones_tensor, twos_tensor, threes_tensor, fours_tensor, fives_tensor, sixes_tensor, sevens_tensor, eights_tensor, nines_tensor]


# In[6]:


def split(dset):
    return train_test_split(dset, test_size=0.2)

def stack_tensors(arr):
    return [torch.stack(i).float() / 255 for i in arr]

def cat(arr):
    train_x = torch.cat(arr).view(-1, 28*28)
    train_y_list = [tensor([i]*len(arr[i])) for i in range(len(arr))]
    
    stacked_y = [x for i in train_y_list for x in i]

    train_x = train_x.reshape(-1, 28, 28)
    train_x = torch.unsqueeze(train_x, 1)
    train_x = torch.tile(train_x, (1, 3, 1, 1))

    return list(zip(train_x, stacked_y))


# In[7]:


batch_size = 64


# In[8]:


stack = stack_tensors(tensors)


# In[9]:


cat_stack = cat(stack)
cat_stack[0]


# In[12]:


train_ds, valid_ds = split(cat_stack)


# In[13]:


train_dl = DataLoader(train_ds, batch_size=batch_size)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)


# In[14]:


dls = DataLoaders(train_dl, valid_dl)


# In[15]:


resnet = resnet18().to(dev)


# In[16]:


trainer = Learner(dls, resnet, loss_func=F.cross_entropy, metrics=error_rate)


# In[17]:


trainer.fit(n_epoch=4, lr=1e-001)


# In[18]:


print('DONE!!')
