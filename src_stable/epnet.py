import random

from matplotlib import pyplot as plt
from individual import Individual
from losses import prechelt_mse_loss
import tqdm
from copy import deepcopy
from utils import rank_selection

class EPNet:
    def __init__(self, X_train, X_val, y_train, y_val,
                 input_dim, output_dim, max_hidden_dim,
                 min_hidden_dim=2, connection_density=0.75,
                 max_mutated_hidden_nodes=2,
                 max_mutated_connections=3,
                 population_size=20, start_epochs=100, training_epochs=100
                 ,criterion=prechelt_mse_loss) -> None:
        # 数据集
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        # 算法参数
        self.training_epochs = training_epochs
        self.max_mutated_hidden_nodes = max_mutated_hidden_nodes
        self.max_mutated_connections = max_mutated_connections
        self.criterion = criterion
        self.start_epochs = start_epochs
        # 统计指标
        self.situation_names = ['bp', 'sa', 'delete_node', 'delete_connection', 'add_node', 'add_connection']
        self.situation_counts = [0 for _ in range(len(self.situation_names))]
        # 构建种群
        self.population = [Individual(input_dim, output_dim, max_hidden_dim,
                                      min_hidden_dim, connection_density)
                           for _ in range(population_size)]
        
    def reset_parameters(self):
        # 初始化训练
        bar = tqdm.tqdm(self.population, desc="Intial BP Trainig", position=0, leave=True, colour='green')
        for individual in bar:
            individual.reset_parameters()
            self.fit_bp(individual, epochs=self.start_epochs)
    def draw_metrics(self, path="test/epnet_metrics.png", type="pie"):
        if type=="pie":
            plt.pie(self.situation_counts, labels=self.situation_names, autopct='%1.1f%%')
        else:
            plt.figure(figsize=(10, 5))
            plt.bar(self.situation_names, self.situation_counts)
        plt.savefig(path)
    
    def try_bp_or_rollback(self, pioneer:Individual, checkpoint)->bool:
        self.fit_bp(pioneer)
        if pioneer.previous_train_success:
            return True # 成功，worst被best的offspring替代
        pioneer.__dict__.update(checkpoint.__dict__) # 丢弃子代；worst继续从parent开始改进。
        return False
    def try_sa_or_rollback(self, pioneer:Individual, checkpoint)->bool:
        pioneer.fit_sa(self.X_train, self.y_train, self.X_val, self.y_val,
                                       epochs_per_temperature=self.training_epochs,max_temperature=5, criterion=self.criterion)
        if pioneer.previous_train_success:
            return True # 成功，worst被best的offspring替代
        pioneer.__dict__.update(checkpoint.__dict__) # 丢弃子代；worst继续从parent开始改进。
        return False
    def fit_bp(self, individual, epochs=None):
        if epochs is None:
            epochs = self.training_epochs
        individual.fit_bp(self.X_train, self.y_train, self.X_val, self.y_val,
                            epochs=epochs, criterion=self.criterion, 
                              position=1, leave=False)
    # def run(self, train_loader, test_loader):
    def run(self, epochs=100,):
        bar = tqdm.tqdm(range(epochs), desc="EPNet", position=0, leave=False, colour='green')
        for i in bar:
            # best_individual = max(
            #     self.population, key=lambda i: i.current_fitness)
            # worst_individual = min(
            #     self.population, key=lambda i: i.current_fitness)
            self.population.sort(key=lambda i:i.current_fitness)
            worst_individual = self.population[0]
            best_individual = self.population[
                rank_selection(len(self.population), k=1)[0]] # 随机选择一个最好的
            bar.set_postfix(best_fitness=best_individual.current_fitness)
            # 1. Hybrid training mutation
            if best_individual.previous_train_success:
                self.fit_bp(best_individual)
                self.situation_counts[0] += 1
                continue  # 因为上一轮成功，不管这一轮是否成功，直接继续训练替代父代。这一轮的不成功是下一次在说。
            # 尝试SA跳出局部最优解
            backup_best_individual = deepcopy(best_individual)
            if self.try_sa_or_rollback(worst_individual, backup_best_individual) : 
                self.situation_counts[1] += 1
                continue 
            # 接下来变异的是 worst_individual
            worst_individual.__dict__.update(backup_best_individual.__dict__)                
            # 2. hidden node deletion
            worst_individual.delete_node(self.max_mutated_hidden_nodes)
            if self.try_bp_or_rollback(worst_individual, backup_best_individual) : 
                self.situation_counts[2] += 1
                continue
            # 3. connection deletion
            worst_individual.delete_connection(self.X_val, self.y_val, prechelt_mse_loss,
                                               max_mutated_connections=self.max_mutated_connections)
            if self.try_bp_or_rollback(worst_individual, backup_best_individual) : 
                self.situation_counts[3] += 1
                continue
            # 4. connection/node addition
            worst_individual.add_connection(self.X_val, self.y_val, prechelt_mse_loss,
                                            self.max_mutated_connections)
            self.fit_bp(worst_individual)
            fitness1 = worst_individual.current_fitness
            checkpoint1 = deepcopy(worst_individual)
            # 退回backup_best_individual, 尝试添加node
            worst_individual.__dict__.update(backup_best_individual.__dict__) 
            worst_individual.add_node(self.max_mutated_hidden_nodes)
            self.fit_bp(worst_individual)
            fitness2 = worst_individual.current_fitness
            if fitness1>fitness2:
                worst_individual.__dict__.update(checkpoint1.__dict__)
                self.situation_counts[4] += 1
            else:
                self.situation_counts[5] += 1
            