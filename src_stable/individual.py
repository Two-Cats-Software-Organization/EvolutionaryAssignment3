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


class Individual(nn.Module):
    """Some Information about Individual"""

    def __init__(self, input_dim, output_dim, max_hidden_dim):
        super(Individual, self).__init__()
        # 1. 形状计算
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_hidden_dim = max_hidden_dim
        self.num_nodes = input_dim+output_dim+max_hidden_dim
        self.num_middle_result_nodes = output_dim+max_hidden_dim
        # 2. 不求导的参数。对BP而言他们是常数
        # - 为了保证to(device)是对的，还是要用Parameter;
        #   为了保证dtype不会不一样，还是要用float32.
        # - i是一种索引方法：从第一个非输入节点（第一个隐藏层节点）开始计算
        #   j是另一种索引方法：从第一个输入节点开始计算。
        # connectivity[j, i] 表示第i个节点的输入是否来自第j个节点
        self.connectivity = nn.Parameter(torch.zeros(self.num_nodes, self.num_middle_result_nodes,
                                                     dtype=torch.int8), requires_grad=False)
        # node_existence[i] 表示第i个节点是否存在。
        self.node_existence = nn.Parameter(torch.zeros(self.num_middle_result_nodes,
                                                       dtype=torch.int8), requires_grad=False)
        self.node_existence[-self.output_dim:] = 1  # 输出节点必须存在
        # 3. 要求导的参数。对BP而言他们是变量。
        # weight[j, i] 表示第i个节点的输入是第j个节点的输出
        self.weight = nn.Parameter(torch.zeros(
            self.num_nodes, self.num_middle_result_nodes))
        # bias[i] 表示第i个节点的bias
        self.bias = nn.Parameter(torch.zeros(self.num_middle_result_nodes))
        # 激活函数
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, input_dim]
        results = torch.empty(x.shape[0], self.num_nodes, device=x.device)
        results[:, :self.input_dim] = x  # 输入节点的输出就是输入
        effective_weight = self.connectivity * self.weight
        for i in range(self.num_middle_result_nodes):
            # 1. effective_weight * 前面所有节点的输出
            res = results[:, :self.input_dim +
                          i] @ effective_weight[:self.input_dim+i, i]
            # 2. 激活
            res = self.activation(res)
            # 3. 加上bias
            res = res + self.bias[i]
            # 4. 乘以node_existence
            res = res * self.node_existence[i]
            # 5. 加到middle_results中
            results[:, self.input_dim+i] = res
        return results[:, -self.output_dim:]

    @lru_cache
    def fitness(self, X_val, y_val, metric=metrics.accuracy_score, use_proba=False):
        with torch.no_grad():
            self.eval()
            y_pred = gpu2numpy(self.forward(X_val))
            if not use_proba:
                y_pred = (y_pred > 0.5).astype(int)
            res = metric(gpu2numpy(y_val), y_pred)
            self.train()
            return res  # Python特性：先会执行finally, 再执行return

# Partial Training by BP
    def fit_bp(self, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor,
               epochs: int = 100, optimizer: optim.Optimizer | None = None,
               criterion: Callable = F.binary_cross_entropy, metric: Callable = metrics.accuracy_score) -> Tuple[bool, np.ndarray]:
        """Partial Training by BP
        Args:
            X_train (torch.Tensor): _description_
            y_train (torch.Tensor): _description_
            X_val (torch.Tensor): _description_
            y_val (torch.Tensor): _description_
            epochs (int, optional): _description_. Defaults to 100.
            optimizer (optim.Optimizer | None, optional): _description_. Defaults to None.
            criterion (Callable, optional): _description_. Defaults to F.binary_cross_entropy.
            metric (Callable, optional): _description_. Defaults to metrics.roc_auc_score.
        Returns:
            Tuple[bool, np.ndarray]: 是否成功; 每一轮的fitness。
        """
        if optimizer is None:
            optimizer = optim.Adam(self.parameters())
        early_stopper = EarlyStopper(patience=epochs//10)
        bar = tqdm.tqdm(range(1, epochs+1), desc="fit_bp")
        # train_losses = torch.zeros(epochs) # 记录每一轮的loss，用于画图
        val_fitness = np.zeros(epochs+1)  # 记录验证集的loss，如果没有下降，说明训练失败。
        val_fitness[0] = self.fitness(X_val, y_val, metric)  # 没有训练之前的loss
        early_stopper.is_continuable(0, val_fitness[0])  # 初始化early_stopper
        for i in bar:
            y_train_pred = self.forward(X_train)
            trian_loss = criterion(y_train_pred.reshape(-1), y_train.to(torch.float32).reshape(-1))
            trian_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            val_fitness[i] = self.fitness(X_val, y_val, metric)
            bar.set_postfix(trian_loss=trian_loss.item(),
                            val_loss=val_fitness[i])
            if not early_stopper.is_continuable(i, val_fitness[i]):
                bar.set_description("fit_bp: early stop")
                break
        return early_stopper.best_score > val_fitness[0], val_fitness


# %%
# if __name__ == "__main__":
indi = Individual(2, 1, 2)
indi.forward(torch.tensor([[1, 2],
                           [3, 4]]))
# %%
torch.autograd.set_detect_anomaly(True)
indi.connectivity[:, :] = 1
nn.init.uniform_(indi.weight, -1, 1)
a= indi.forward(torch.tensor([[1, 2],
                           [3, 4]]))
a.mean().backward()
# %%
from datasets import NParity
from utils import dataset2numpy, numpy2gpu
from sklearn.model_selection import train_test_split
dataset = NParity(2) # XOR 问题
X, Y = dataset2numpy(dataset)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
X_train, X_val, y_train, y_val = numpy2gpu(X_train, X_val, y_train, y_val)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
indi = indi.to(device)
indi.fit_bp(X_train, y_train, X_val, y_val)

# %%
