#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import NParity
from utils import dataset2numpy, numpy2gpu
from sklearn.model_selection import train_test_split
N = 8
dataset = NParity(N) # XOR 问题
X, Y = dataset2numpy(dataset)
# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
X_train, X_val, y_train, y_val = X, X, Y, Y
X_train, X_val, y_train, y_val = numpy2gpu(X_train, X_val, y_train, y_val)
#%%
from epnet import EPNet
net = EPNet(X_train, X_val, y_train, y_val,
            input_dim=N, output_dim=1, max_hidden_dim=N,
            criterion=F.binary_cross_entropy, lr=0.2
            ,population_size=20)
#%%
net.reset_parameters()
#%%
net.run(20)
#%%
metrics = map(lambda i: i.metrics(), net.population)
import pandas as pd
df = pd.DataFrame(metrics)
df.head()
#%%
df.describe()
#%%
best_individual = max(
            # net.population, key=lambda i: i.hidden_nodes()*i.current_fitness)
            # net.population, key=lambda i: i.hidden_nodes())
            net.population, key=lambda i: i.current_fitness)
best_individual.current_fitness
#%%
from sklearn import metrics
best_individual.fitness_sklearn(X_train, y_train, metrics.accuracy_score, False)

#%%
list(best_individual.parameters())

#%%
best_individual.savefig(f"fig/indi-{N}.dot")

#%%
#%%
# best_individual.fit_bp(X_train, y_train, X_val, y_val, epochs=100)
best_individual.fit_bp(X_train, y_train, 
                       X_val, y_val, epochs=1000,
                       optimizer = optim.AdamW(
                           best_individual.parameters(),
                           lr=0.1),)
#%%
best_individual.fit_sa(X_train, y_train, X_val, y_val, 
                      )

#%%
best_individual.metrics()
# %%
from matplotlib import pyplot as plt
plt.style.use('default')
net.draw_metrics(path="fig/epnet_metrics.png", type='bar')
# %%
