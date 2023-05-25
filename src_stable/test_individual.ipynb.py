# %%
from typing import Callable, List, Tuple
from utils import EarlyStopper, gpu2numpy
import torch.nn.init
from functools import lru_cache
import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import sklearn.metrics as metrics
from sklearnex import patch_sklearn
patch_sklearn()
import math
from individual import Individual

import torch.backends
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
# indi = Individual(2, 1, 200)
# indi = Individual(2, 1, 2)
indi = Individual(2, 1, 4)
# indi = torch.compile(indi, mode="max-autotune")
out = indi.forward(torch.tensor([[1, 2],
                           [3, 4]]))
out
#%%

# from torchviz import make_dot,make_dot_from_trace
# graph=make_dot(out,params=dict(indi.named_parameters()),)
# graph.render(filename='indi',view=False,format='png')

#%%
# from tensorwatch import draw_model
# draw_model(indi, [1, 2])
x=torch.randn((1,2))
y=torch.randn((1, 1))
input_names = ["input_0","input_1"]
output_names = ["xor_output"]
# torch.onnx.export(indi,(x,y),'indi.onnx',input_names=input_names,output_names=output_names,
torch.onnx.export(indi,(x,),'indi.onnx',input_names=input_names,output_names=output_names,
  dynamic_axes={'input_0':[0],'output_0':[0]} )

#%%
from objprint import objprint
objprint(indi)
from copy import deepcopy
copy_indi = deepcopy(indi)
#%%
list(indi.parameters()), list(copy_indi.parameters())

# %%
# torch.autograd.set_detect_anomaly(True)
# indi.connectivity[:, :] = 1
# indi.node_existence[:] = 1
nn.init.uniform_(indi.weight, -1, 1)
a= indi.forward(torch.tensor([[1, 2],
                           [3, 4]]))
a.mean().backward()
#%%
indi.bias.grad
#%%
optimizer = optim.Adam(indi.parameters())
optimizer.step()
# %%
from datasets import NParity
from utils import dataset2numpy, numpy2gpu
from sklearn.model_selection import train_test_split
dataset = NParity(2) # XOR 问题
X, Y = dataset2numpy(dataset)
# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
X_train, X_val, y_train, y_val = X, X, Y, Y
X_train, X_val, y_train, y_val = numpy2gpu(X_train, X_val, y_train, y_val)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%%
indi.reset_parameters()
indi


#%%
indi = indi.to(device)
indi.fit_bp(X_train, y_train, X_val, y_val, epochs=100, 
            optimizer=optim.Adam(indi.parameters(), lr=0.01))
            # optimizer=optim.Adam(indi.parameters(), lr=0.5))
            # optimizer=optim.Adam(indi.parameters(), lr=0.01))
# indi.fit_bp(X_train, y_train, X_val, y_val, epochs=100, 
#             )
#%%
# list(indi.parameters())
indi.bias

# %%
indi.fitness(X_val, y_val)
indi.fitness(X_train, y_train, metrics.accuracy_score, False)

# %%
indi(X_train), y_train
