#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')
get_ipython().system('pip install -Uqq fastbook')
import fastbook
fastbook.setup_book()


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


# In[10]:


plt.imshow(cat_stack[1][0].permute(1, 2, 0))


# In[11]:


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


test_zeroes = (path/'testing'/'0').ls().sorted()
test_ones = (path/'testing'/'1').ls().sorted()
test_twos = (path/'testing'/'2').ls().sorted()
test_threes = (path/'testing'/'3').ls().sorted()
test_fours = (path/'testing'/'4').ls().sorted()
test_fives = (path/'testing'/'5').ls().sorted()
test_sixes = (path/'testing'/'6').ls().sorted()
test_sevens = (path/'testing'/'7').ls().sorted()
test_eights = (path/'testing'/'8').ls().sorted()
test_nines = (path/'testing'/'9').ls().sorted()

test_zeroes_tensor = [tensor(Image.open(o)) for o in test_zeroes]
test_ones_tensor = [tensor(Image.open(o)) for o in test_ones]
test_twos_tensor = [tensor(Image.open(o)) for o in test_twos]
test_threes_tensor = [tensor(Image.open(o)) for o in test_threes]
test_fours_tensor = [tensor(Image.open(o)) for o in test_fours]
test_fives_tensor = [tensor(Image.open(o)) for o in test_fives]
test_sixes_tensor = [tensor(Image.open(o)) for o in test_sixes]
test_sevens_tensor = [tensor(Image.open(o)) for o in test_sevens]
test_eights_tensor = [tensor(Image.open(o)) for o in test_eights]
test_nines_tensor = [tensor(Image.open(o)) for o in test_nines]


def test_stack_tensors(arr):
    return [torch.stack(i).float() / 255 for i in arr]

def test_cat(arr):
    train_x = torch.cat(arr).view(-1, 28*28)
    train_y_list = [tensor([i]*len(arr[i])) for i in range(len(arr))]
    
    stacked_y = [x for i in train_y_list for x in i]

    train_x = train_x.reshape(-1, 28, 28)
    train_x = torch.unsqueeze(train_x, 1)
    train_x = torch.tile(train_x, (1, 3, 1, 1))

    return train_x, stacked_y


# In[19]:


test_tensors = [test_zeroes_tensor, 
           test_ones_tensor, 
           test_twos_tensor, 
           test_threes_tensor, 
           test_fours_tensor, 
           test_fives_tensor, 
           test_sixes_tensor, 
           test_sevens_tensor, 
           test_eights_tensor, 
           test_nines_tensor
          ]


# In[20]:


PATH = "models/mnist_model.pth"


# In[21]:


torch.save(resnet.state_dict(), PATH)


# In[22]:


test_stack = test_stack_tensors(test_tensors)


# In[23]:


test_ds = test_cat(test_stack)


# In[24]:


model = resnet18()
model.load_state_dict(torch.load(PATH))


# In[25]:


preds = model(test_ds[0]).to(dev)


# In[26]:


get_ipython().run_line_magic('pinfo', 'torch.argmax')


# In[27]:


valid_tensors = torch.LongTensor(test_ds[1]).to(dev)


# In[28]:


torch.argmax(preds[4000:4100], dim=1)


# In[29]:


valid_tensors[4000:4100] 


# In[30]:


test_ds[1][4000:4100]


# In[31]:


wrong = valid_tensors[0] != torch.argmax(preds, dim=1)


# In[32]:


preds[0].shape, valid_tensors.shape


# In[33]:


wrong.shape

