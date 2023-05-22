#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        self.node_existence[-self.output_dim:] = 1 # 输出节点必须存在
        # 3. 要求导的参数。对BP而言他们是变量。
        # weight[j, i] 表示第i个节点的输入是第j个节点的输出
        self.weight = nn.Parameter(torch.zeros(self.num_nodes, self.num_middle_result_nodes))
        # bias[i] 表示第i个节点的bias
        self.bias = nn.Parameter(torch.zeros(self.num_middle_result_nodes))
        # 激活函数
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # x: [batch_size, input_dim]
        results = torch.empty(x.shape[0], self.num_nodes)
        results[:, :self.input_dim] = x # 输入节点的输出就是输入
        effective_weight = self.connectivity * self.weight
        for i in range(self.num_middle_result_nodes):
            # 1. effective_weight * 前面所有节点的输出
            res = results[:, :self.input_dim+i] @ effective_weight[:self.input_dim+i, i]
            # 2. 激活
            res = self.activation(res)
            # 3. 加上bias
            res = res + self.bias[i]
            # 4. 乘以node_existence
            res = res * self.node_existence[i]
            # 5. 加到middle_results中
            results[:, self.input_dim+i] = res
        return results[:, -self.output_dim:]
    
    
#%%
# if __name__ == "__main__":
import torch.nn.init
indi = Individual(2, 3, 2)
indi.forward(torch.tensor([[1, 2], 
                        [3, 4]]))
#%%
indi.connectivity[:, :] = 1
nn.init.uniform_(indi.weight, -1, 1)
indi.forward(torch.tensor([[1, 2], 
                        [3, 4]]))
# %%
