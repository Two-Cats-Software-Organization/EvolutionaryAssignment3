#%%
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

indi = Individual(2, 1, 4)
# indi = torch.compile(indi, mode="max-autotune")
out = indi.forward(torch.tensor([[1, 2],
                           [3, 4]]))

list(indi.parameters())
# %%
indi.reset_parameters()
list(indi.parameters())
# %%
success = indi.delete_node(2)
list(indi.parameters()), success
# %%
success = indi.add_node(2)
list(indi.parameters()), success
# %%
from datasets import NParity
from utils import dataset2numpy, numpy2gpu
from sklearn.model_selection import train_test_split
dataset = NParity(2) # XOR 问题
X, Y = dataset2numpy(dataset)
# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
X_train, X_val, y_train, y_val = X, X, Y, Y
X_train, X_val, y_train, y_val = numpy2gpu(X_train, X_val, y_train, y_val)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# %%
indi = indi.to(device)
indi.fit_bp(X_train, y_train, X_val, y_val, epochs=100)
#%%
indi.connections()
# %%
from losses import prechelt_mse_loss
# indi.connection_importance(X_val, y_val, prechelt_mse_loss)
# indi.connection_importance(X_val, y_val, F.mse_loss)
# indi.connection_importance(X_val, y_val, F.binary_cross_entropy)
importance = indi.connection_importance_prob(X_val, y_val, prechelt_mse_loss)
importance
#%%
(importance/importance.sum()).sum()
(importance/importance.sum()).sum()
(1-importance).sum()
# 1-importance
#%%
indi.delete_connection(X_val, y_val)

#%%
indi.add_connection(X_val, y_val)
#%%
# indi.connectivity
# indi.connectivity[[torch.tensor([2, 3]), 
#                    torch.tensor([1, 0]), 
#                     torch.tensor([0, 0])]] = 1
# %%
effective_weights = indi.weight*indi.connectivity.triu(
    diagonal=-indi.input_dim+1)*torch.unsqueeze(indi.node_existence, 0)
effective_weights
# %%
