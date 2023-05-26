# %%
import bisect
import sklearn.metrics as metrics
from losses import prechelt_mse_loss
from sko.SA import SA
import math
import random
from typing import Callable, List, Tuple
from utils import EarlyStopper, gpu2numpy, numpy2gpu
import torch.nn.init
from functools import lru_cache
import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import networkx as nx
from sklearnex import patch_sklearn
patch_sklearn()


class Individual(nn.Module):
    """Some Information about Individual"""

    def __init__(self, input_dim, output_dim, max_hidden_dim, min_hidden_dim=2, connection_density=0.75):
        super(Individual, self).__init__()
        assert max_hidden_dim >= min_hidden_dim
        # 0. 演化标记
        self.previous_train_success = False
        self.current_fitness = 0
        self.epoches_since_structured = 0
        # 1. 形状计算
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_hidden_dim = max_hidden_dim
        self.min_hidden_dim = min_hidden_dim
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
        self.reset_parameters(connection_density)

    def savefig(self, path="test/indi.dot"):
        g = self.to_networkx()
        nx.drawing.nx_pydot.write_dot(g,path) 
    def to_networkx(self):
        G = nx.DiGraph()
        # nodes. 统一使用j索引；命名时按照节点的类型，重新计数，称为k索引
        G.add_node(-1, label="1", color="#2ecc71", fontcolor="#2ecc71",
                   shape="circle", style="bold")  # bias
        nodes = [(j, {"color": "#2ecc71", "style": "wedged", "fontcolor": "#2ecc71", "shape": "circle",
                               "label": f"x{j}"}) for j in range(self.input_dim)]
        k = 0
        for i in list(self.node_existence[:-self.output_dim].nonzero()):
            j = i.item()+self.input_dim
            nodes.append((j, {"shape": "circle", "style": "wedged",
                              "color": "#3498db", "fontcolor": "#3498db",
                              "label": f"z{k}", "bias": str(self.bias[i].item())}))
            k += 1
        for k in range(self.output_dim):
            i = self.max_hidden_dim+k
            j = self.input_dim+i
            nodes.append((j, {"shape": "circle", "style": "wedged",
                              "color": "#e74c3c", "fontcolor": "#e74c3c",
                              "label": f"y{k}", "bias": str(self.bias[i].item())}))
        G.add_nodes_from(nodes)
        # edges. 基于j索引，寻找有效连接，把weight绑上去。
        # edges = []
        for i in range(self.connectivity.shape[1]):
            for j in range(self.connectivity.shape[0]):
                if j >= i+self.input_dim:
                    break
                if self.connectivity[j][i] and self.node_existence[i]:
                    if j >= self.input_dim and self.node_existence[j-self.input_dim] == False:
                        continue
                    # edges.append((j, i+self.input_dim, self.weight[j, i].item()))
                    weight = self.weight[j, i].item()
                    G.add_edge(j, i+self.input_dim,
                               weight=weight,
                               label=f"{weight:.2f}",
                               fontcolor="steelblue" if i < self.max_hidden_dim else "goldenrod"
                               )
        # bias

        for i in list(self.node_existence.nonzero()):
            i = i.item()
            j = i+self.input_dim
            bias = self.bias[i].item()
            G.add_edge(-1, j,
                       bias=bias,
                       label=f"{bias:.2f}",
                       fontcolor="steelblue" if i < self.max_hidden_dim else "goldenrod"
                       )

        # G.add_weighted_edges_from(edges)
        return G

    def reset_parameters(self, connection_density=0.75) -> None:
        self.connectivity = nn.Parameter(torch.nn.init.uniform_(torch.zeros_like(self.connectivity)
                                                                .to(torch.float32), 0, 1) < connection_density,
                                         requires_grad=False)

        # initial_nodes = random.choices(range(self.max_hidden_dim), k=random.randint(
        initial_nodes = random.sample(range(self.max_hidden_dim), k=random.randint(
            self.min_hidden_dim, self.max_hidden_dim))
        self.node_existence[:] = 0
        self.node_existence[initial_nodes] = 1
        self.node_existence[-self.output_dim:] = 1
        for i in range(self.num_middle_result_nodes):
            fan_in = i+self.input_dim
            v_squared = 1/(fan_in*(0.25**2)*(1+0.5**2))
            torch.nn.init.normal_(self.weight[:, i], 0, math.sqrt(v_squared))
        torch.nn.init.constant_(self.bias, 0)

        # torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, effective_weight=None):
        # x: [batch_size, input_dim]
        # results = [torch.empty(self.num_nodes, device=x.device) for _ in range(x.shape[0])]
        results = []
        results += list(x.T.to(torch.float32))  # 输入节点的输出就是输入re
        if effective_weight is None:
            effective_weight = self.connectivity * self.weight
        for i in range(self.num_middle_result_nodes):
            # 1. effective_weight * 前面所有节点的输出
            # previous_signals = results[:, :self.input_dim +
            #               i]
            previous_signals = torch.vstack(results[:self.input_dim+i]).T
            connections_weights = effective_weight[:self.input_dim+i, i]
            # res = previous_signals @ connections_weights
            res = torch.matmul(previous_signals, connections_weights)
            # 2. 加上bias
            res = res + self.bias[i]
            # 3. 激活
            # res = self.activation(res.clone())
            res = self.activation(res)
            # 4. 乘以node_existence
            res = res * self.node_existence[i]
            # 5. 加到middle_results中
            results.append(res)
        return torch.vstack(results[-self.output_dim:]).T

    # @lru_cache
    def fitness_sklearn(self, X_val, y_val, metric=metrics.accuracy_score, use_proba=False):
        # def fitness(self, X_val, y_val, metric=lambda a,b:-(metrics.log_loss(a, b)), use_proba=True):
        with torch.no_grad():
            self.eval()
            y_pred = gpu2numpy(self.forward(X_val))
            if not use_proba:
                y_pred = (y_pred > 0.5).astype(int)
            res = metric(gpu2numpy(y_val), y_pred)  # sklearn 风格，true在前
            self.train()
            return res  # Python特性：先会执行finally, 再执行return

    def fitness_torch(self, X_val, y_val, metric=F.binary_cross_entropy):
        with torch.no_grad():
            self.eval()
            y_pred = self.forward(X_val)
            # print(y_pred.max(), y_pred.min())
            # pytorch 风格，y_pred在前
            res = -metric(y_pred.reshape(-1),
                          y_val.to(torch.float32).reshape(-1))
            self.train()
            return res.item()  # Python特性：先会执行finally, 再执行return

    def fit_bp(self, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor,
               epochs: int = 100, optimizer: optim.Optimizer | None = None,
               criterion: Callable = F.binary_cross_entropy, 
               position=0, leave=True) -> Tuple[bool, np.ndarray]:
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
        initial_val_fitness = self.fitness_torch(X_val, y_val, criterion)
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=3e-4)
        # early_stopper = EarlyStopper(patience=epochs//10)
        early_stopper = EarlyStopper(patience=epochs//1)
        # early_stopper = EarlyStopper(patience=epochs//2)
        bar = tqdm.tqdm(range(1, epochs+1), desc="fit_bp", position=position, leave=leave, colour='yellow')
        # val_fitness # 记录验证集的loss，如果没有下降，说明训练失败。
        train_fitness = np.zeros(epochs+1)
        train_fitness[0] = self.fitness_torch(
            X_train, y_train, criterion)  # 没有训练之前的loss
        # 针对训练集的stopper。只是把best state留下来

        # 一开始state不要留，让外面而不是里面恢复
        # early_stopper.is_continuable(
        #     0, train_fitness[0], self.state_dict())  # 初始化early_stopper
        for i in bar:
            self.epoches_since_structured += 1
            y_train_pred = self.forward(X_train)
            # pytorch 风格，y_pred在前
            trian_loss = criterion(
                y_train_pred.reshape(-1), y_train.to(torch.float32).reshape(-1))
            trian_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_fitness[i] = self.fitness_torch(X_train, y_train, criterion)
            bar.set_postfix(trian_loss=trian_loss.item(),
                            val_fitness=train_fitness[i])
            if not early_stopper.is_continuable(i, train_fitness[i], self.state_dict()):
                bar.set_description("fit_bp: early stop")
                break
        self.load_state_dict(early_stopper.best_state_dict)
        final_fitness = self.fitness_torch(X_val, y_val, criterion)
        self.current_fitness = final_fitness
        self.previous_train_success = initial_val_fitness < final_fitness
         
        # return early_stopper.best_score > train_fitness[0], train_fitness
        return self.previous_train_success, train_fitness

    def fit_sa(self, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor,
               epochs_per_temperature: int = 100, max_temperature=5,
               criterion: Callable = F.binary_cross_entropy) -> Tuple[bool, np.ndarray]:
        # old_state_dict = self.state_dict() # 在外面做保存，里面训练了就是训练了
        initial_val_fitness = self.fitness_torch(X_val, y_val, criterion)
        dims = self.weight.numel() + self.bias.numel()
        x0 = gpu2numpy(torch.hstack(
            (self.weight.flatten(), self.bias.flatten())))

        def objective(x: np.ndarray):
            q = numpy2gpu(x, device=self.weight.device)
            self.weight = nn.Parameter(
                q[:self.weight.numel()].reshape(self.weight.shape))
            self.bias = nn.Parameter(
                q[self.weight.numel():].reshape(self.bias.shape))
            # SA 是求最小值的，所以要加负号
            return -self.fitness_torch(X_train, y_train, criterion)
        optimizer = SA(func=objective, x0=x0,
                       T_max=max_temperature, T_min=1e-7, L=epochs_per_temperature,)
        best_x, best_y = optimizer.run()
        self.epoches_since_structured += len(optimizer.best_y_history)
        objective(best_x)  # 将状态设置为最佳状态
        final_fitness = self.fitness_torch(X_val, y_val, criterion)
        self.current_fitness = final_fitness
        self.previous_train_success = initial_val_fitness < final_fitness
        return self.previous_train_success, np.array(optimizer.best_y_history)

    def delete_node(self, max_mutated_hidden_nodes:int=2)->int:
        """删除q个隐层节点， 其中q服从[1, min(max_mutated_hidden_nodes, self.hidden_nodes())]的均匀分布。
        Args:
            max_mutated_hidden_nodes (int, optional): _description_. Defaults to 2.
        Returns:
            int: 成功删除的节点数量。如果没有已经没有节点可以被删除，返回0。
        """
        self.epoches_since_structured = 0 # 修改了网络结构，所以数据归零
        hidden_nodes = self.hidden_nodes()
        if hidden_nodes<=0:
            return 0
        q = random.randint(1, min(max_mutated_hidden_nodes, hidden_nodes))
        nodes = [i.item() for i in self.node_existence[:-self.output_dim].nonzero()]
        deletes = random.sample(nodes, q) # sample 和 chooses区别：不会重复，一定要删除那么多节点。
        self.node_existence[deletes] = 0
        return q

    def connection_importance(self, X:torch.Tensor, y:torch.Tensor, criterion:Callable=F.binary_cross_entropy)->torch.Tensor:
        """计算每条连接的重要性，用于删除连接。
        Args:
            X (torch.Tensor): 训练集的输入
            y (torch.Tensor): 训练集的输出
            criterion (Callable, optional): 损失函数. Defaults to F.binary_cross_entropy.
        Returns:
            torch.Tensor: 每条连接的重要性
        """
        self.epoches_since_structured = 0
        y_pred = self.forward(X)
        grads = torch.empty((len(y_pred), *self.weight.shape))
        for i in range(len(y_pred)):
            self.zero_grad()
            loss = criterion((y_pred[i]).reshape(-1), y[i].to(torch.float32).reshape(-1))
            loss.backward(retain_graph=True)
            grads[i] = self.weight.grad      
        effective_weights = self.weight*self.connectivity.triu(
            diagonal=-self.input_dim+1)*torch.unsqueeze(self.node_existence, 0)
        importance = torch.unsqueeze(effective_weights, 0)+grads
        importance = torch.mean(importance, dim=0)/torch.std(importance, dim=0)/math.sqrt(len(y_pred))
        return importance
        
    def delete_connection(self, X:torch.Tensor, y:torch.Tensor, criterion:Callable=F.binary_cross_entropy, 
                           max_mutated_connections=3):
        self.epoches_since_structured = 0
        connections = self.connections()
        if connections<=0:
            return 0
        q = random.randint(1, min(max_mutated_connections, connections))
        importance = 1/self.connection_importance(X, y, criterion)
        importance = torch.nan_to_num(importance, nan=0, posinf=10, neginf=-10).reshape(-1)
        importance = importance/torch.sum(importance)
        if torch.isnan(importance).sum()!=0: return 0
        importance = [i.item() for i in importance]
        deletes = random.choices(range(len(importance)), weights=importance, k=q)
        deletes = list({(i//self.connectivity.shape[1], i%self.connectivity.shape[1]) for i in deletes})
        for delete in deletes:
            self.connectivity[delete] = 0
        # self.connectivity[deletes] = 0
        return q
        

    def add_node(self, max_mutated_hidden_nodes=2, alpha=0.25)->int:
        self.epoches_since_structured = 0        
        available_new_nodes = [i.item() for i in (1-self.node_existence[:-self.output_dim]).nonzero()]
        if len(available_new_nodes)<=0:
            return 0
        q = random.randint(1, min(max_mutated_hidden_nodes, len(available_new_nodes)))
        # nodes = [i.item() for i in self.node_existence[:-self.output_dim].nonzero()]
        nodes = [i.item() for i in self.node_existence[:].nonzero()]
        offsprings = random.sample(available_new_nodes, q)
        for offspring in offsprings:
            insert_place = max(0, bisect.bisect(nodes, offspring)-1) # 优先选择左边的存在节点，如果不行再选择右边。
            parent = nodes[insert_place]
            self.cell_division(offspring, parent, alpha=alpha)
        return q
    def cell_division(self, offspring, parent, alpha=0.25):
        # 输入的都是i索引。
        # offspring还不存在，parent存在。
        bigger_one = max(offspring, parent)
        smaller_one = min(offspring, parent)
        with torch.no_grad():
            self.node_existence[offspring] = 1
            self.weight[offspring, parent] = 0
            # 1. 前面的权重；不需要考虑 i>=j 因为是冗余的参数而已。
            self.weight[:smaller_one, offspring] = self.weight[:smaller_one, parent]
            self.bias[offspring] = self.bias[parent] 
            # 2. 后面的权重
            for i in range(bigger_one+1, self.num_middle_result_nodes):
                if self.node_existence[i]==1:
                    self.weight[offspring, i] = self.weight[parent, i] * (1+alpha)
                    self.weight[parent, i] = self.weight[parent, i] * (-alpha)
         
        

    def add_connection(self, X:torch.Tensor, y:torch.Tensor, criterion:Callable=F.binary_cross_entropy, 
                       max_mutated_connections=3):
        # TODO 在连接没有恢复之前不可能求导。
        self.epoches_since_structured = 0
        # 1. 确定q
        possible_connections = torch.ones_like(
            self.connectivity).triu(
                diagonal=-self.input_dim+1)*torch.unsqueeze(self.node_existence, 0)
        current_connectivity = self.real_connectivity()
        resumable_connections = possible_connections & (~current_connectivity)
        num_resumable_connections = int(resumable_connections.sum().item())
        if num_resumable_connections<=0:
            return 0
        q = random.randint(1, min(max_mutated_connections, num_resumable_connections))
        # 2. 确定增加
        resumable_connections = [index for index in resumable_connections.nonzero()]
        adds = random.sample(resumable_connections, k=q)
        for add in adds:
            self.connectivity[add] = 1
        return q

    # 一些指标
    def hidden_nodes(self):
        return int(self.node_existence[:-self.output_dim].sum().item())

    def real_connectivity(self):
        return (self.connectivity.triu(diagonal=-self.input_dim+1)
                    *torch.unsqueeze(self.node_existence, 0))
    def connections(self):
        return int(self.real_connectivity().sum().item())

    def metrics(self):
        return dict(
            hidden_nodes=self.hidden_nodes(),
            connections=self.connections(),
            epoches_since_structured=self.epoches_since_structured,
            fitness=self.current_fitness
        )

# %%
