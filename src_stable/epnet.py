import random
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
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.training_epochs = training_epochs
        self.max_mutated_hidden_nodes = max_mutated_hidden_nodes
        self.max_mutated_connections = max_mutated_connections
        self.criterion = criterion

        self.population = [Individual(input_dim, output_dim, max_hidden_dim,
                                      min_hidden_dim, connection_density)
                           for _ in range(population_size)]
        for individual in self.population:
            individual.fit_bp(self.X_train, self.y_train, self.X_val, self.y_val,
                              epochs=start_epochs, criterion=prechelt_mse_loss)

    
    
    def try_bp_or_rollback(self, pioneer:Individual)->bool:
        old_state_dict = pioneer.state_dict()
        pioneer.fit_bp(self.X_train, self.y_train, self.X_val, self.y_val,
                                       epochs=self.training_epochs, criterion=self.criterion)
        if pioneer.previous_train_success:
            return True # 成功，worst被best的offspring替代
        pioneer.load_state_dict(old_state_dict) # 丢弃子代；worst继续从parent开始改进。
        return False
    # def run(self, train_loader, test_loader):
    def run(self, epochs=100,):
        bar = tqdm.tqdm(range(epochs))
        for i in bar:
            # best_individual = max(
            #     self.population, key=lambda i: i.current_fitness)
            # worst_individual = min(
            #     self.population, key=lambda i: i.current_fitness)
            self.population.sort(key=lambda i:i.current_fitness)
            worst_individual = self.population[0]
            best_individual = self.population[
                rank_selection(len(self.population), k=1)[0]] # 随机选择一个最好的
            # 1. Hybrid training mutation
            if best_individual.previous_train_success:
                best_individual.fit_bp(self.X_train, self.y_train, self.X_val, self.y_val,
                                       epochs=self.training_epochs, criterion=self.criterion)
                continue  # 因为上一轮成功，不管这一轮是否成功，直接继续训练替代父代。这一轮的不成功是下一次在说。
            
            old_state_dict = best_individual.state_dict()
            best_individual.fit_sa(self.X_train, self.y_train, self.X_val, self.y_val,
                                    epochs_per_temperature=100, max_temperature=5, criterion=self.criterion
                                    )
            if best_individual.previous_train_success:
                continue  # 成功的退火，替代父代。
            best_individual.load_state_dict(
                old_state_dict)  # 子代没用，保留parent。
            # 接下来修改的是 worst_individual
            # worst_state_dict = worst_individual.state_dict()
            worst_individual.load_state_dict(best_individual.state_dict())                
            # 2. hidden node deletion
            worst_individual.delete_node(self.max_mutated_hidden_nodes)
            if self.try_bp_or_rollback(worst_individual) : continue
            # 3. connection deletion
            worst_individual.delete_connection(self.max_mutated_connections)
            if self.try_bp_or_rollback(worst_individual) : continue
            # 4. connection/node addition
            worst_individual.add_connection(self.max_mutated_connections)
            worst_individual.fit_bp(self.X_train, self.y_train, self.X_val, self.y_val,
                                       epochs=self.training_epochs, criterion=self.criterion)
            fitness1 = worst_individual.current_fitness
            state_dict1 = worst_individual.state_dict()
            
            worst_individual.load_state_dict(best_individual.state_dict())
            worst_individual.add_node(self.max_mutated_hidden_nodes)
            worst_individual.fit_bp(self.X_train, self.y_train, self.X_val, self.y_val,
                                       epochs=self.training_epochs, criterion=self.criterion)
            fitness2 = worst_individual.current_fitness
            
            if fitness1>fitness2:
                worst_individual.load_state_dict(state_dict1)
            
            