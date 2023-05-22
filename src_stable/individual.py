import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Individual(nn.Module):
    """Some Information about Individual"""
    def __init__(self, input_dim, output_dim, max_hidden_dim):
        super(Individual, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_hidden_dim = max_hidden_dim
        self.num_nodes = input_dim+output_dim+max_hidden_dim
        self.num_middle_result_nodes = output_dim+max_hidden_dim
        # 不求导，对BP而言是常数
        # 为了保证to(device)是对的，还是要用Parameter;
        # 为了保证dtype不会不一样，还是要用float32.
        self.connectivity = nn.Parameter(torch.zeros(self.num_nodes, self.num_middle_result_nodes,
                                        dtype=torch.float32), requires_grad=False)
        self.node_existence = nn.Parameter(torch.zeros(self.max_hidden_dim, 
                                                       dtype=torch.float32), requires_grad=False)
        # 求导
        # weight[j, i] 表示第i个节点的输入是第j个节点的输出
        self.weight = nn.Parameter(torch.zeros(self.num_nodes, self.num_middle_result_nodes))
        self.bias = nn.Parameter(torch.zeros(self.num_middle_result_nodes))
        # 激活函数
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # x: [batch_size, input_dim]
        middle_results = torch.zeros(x.shape[0], self.num_middle_result_nodes)
        middle_results[:, :self.input_dim] = x
        effective_weight = self.connectivity * self.weight
        for i in range(self.num_middle_result_nodes):
            # 1. effective_weight * 前面所有节点的输出
            res = middle_results[:, :self.input_dim+i] @ effective_weight[:self.input_dim+i, i]
            # 2. 激活
            res = self.activation(res)
            # 3. 加上bias
            res = res + self.bias[i]
            # 4. 乘以node_existence
            res = res * self.node_existence[i]
            # 5. 加到middle_results中
            middle_results[:, i] = res
        return middle_results[:, -self.output_dim:]