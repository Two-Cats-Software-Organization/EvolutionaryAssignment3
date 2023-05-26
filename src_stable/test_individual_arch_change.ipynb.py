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
