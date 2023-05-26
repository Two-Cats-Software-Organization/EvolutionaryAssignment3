#%%
from datasets import NParity
from utils import dataset2numpy, numpy2gpu
from sklearn.model_selection import train_test_split
dataset = NParity(2) # XOR 问题
X, Y = dataset2numpy(dataset)
# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
X_train, X_val, y_train, y_val = X, X, Y, Y
X_train, X_val, y_train, y_val = numpy2gpu(X_train, X_val, y_train, y_val)
#%%
from epnet import EPNet
net = EPNet(X_train, X_val, y_train, y_val,
            input_dim=2, output_dim=1, max_hidden_dim=2)
#%%
net.reset_parameters()
#%%
net.population[0].fit_bp(X_train, y_train, X_val, y_val, epochs=100)
# %%
from matplotlib import pyplot as plt
plt.style.use('default')
net.situation_counts = [10, 10, 10, 20, 30, 90]
# net.draw_metrics()
net.draw_metrics(type='bar')
# %%
