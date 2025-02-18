
import numpy as np
import random
import math
import time
import copy
import vrplib

# Classe da metaheurística
class ALNS:
    
    def __init__(self, feasible_solution: None, evaluation_foo = None, 
                 destroy_methods: list = None, repair_methods: list = None, 
                 r: int = 0.1, sigma_values: list[int] = [], methods_hyperparameters: dict = None, 
                 solution_generator = None):
        
        assert(destroy_methods is not None)
        assert(repair_methods is not None)
        assert(evaluation_foo is not None)
        assert(feasible_solution is not None)
        assert(methods_hyperparameters is not None)
        assert(solution_generator is not None)
        
        self.methods_hyperparameters = methods_hyperparameters
        self.solution_generator = solution_generator
            
        self.evaluation_function = evaluation_foo
        self.initial_solution = copy.deepcopy(feasible_solution)
        
        self.best_solution = copy.deepcopy(feasible_solution)
        self.best_cost = self.evaluation_function(self.best_solution) 
        self.time_to_best = None
        
        self.sigma_values = sigma_values
        self.r = r
        
        self.destroy_methods = destroy_methods
        self.repair_methods = repair_methods
        self.costs_per_iter = []
    
        self.__init_destroy_repair_methods_metadata()
    
    #q_interval
    def __init_destroy_repair_methods_metadata(self):
        
        ## iniciando vetores de peso, score, e de quantidades de utilização
        self.repair_weights = [1]*(len(self.repair_methods)) 
        self.destroy_weights = [1]*(len(self.destroy_methods))
        self.repair_scores = [0]*(len(self.repair_methods))
        self.destroy_scores = [0]*(len(self.destroy_methods))
        self.repair_n_times_used = [0]*(len(self.repair_methods))
        self.destroy_n_times_used = [0]*(len(self.destroy_methods))
        
    ## atualiza peso dos métodos
    def __update_methods_weights(self) -> None:
        
        min_weight = 1e-12 ## definindo um valor mínimo de weight para impedir que diminua indefinidamente
        
        ## atualizando pesos dos métodos de destruição
        for i in range(len(self.destroy_methods)):
            if self.destroy_n_times_used[i] > 0:
                self.destroy_weights[i] = self.destroy_weights[i]*(1 - self.r) + self.r * (self.destroy_scores[i] / self.destroy_n_times_used[i])
                self.destroy_weights[i] = max(min_weight, self.destroy_weights[i])
            self.destroy_n_times_used[i] = 0
            self.destroy_scores[i] = 0 

        ## atualizando pesos dos métodos de reparação
        for i in range(len(self.repair_methods)):
            if self.repair_n_times_used[i] > 0:
                self.repair_weights[i] = self.repair_weights[i]*(1 - self.r) + self.r * (self.repair_scores[i] / self.repair_n_times_used[i])
                self.repair_weights[i] = max(min_weight, self.repair_weights[i])
            self.repair_n_times_used[i] = 0
            self.repair_scores[i] = 0
            
        return 
    
    ## Seleciona um método de destruição e um de reparação aleatóriamente e retorna o índice dos métodos
    def __select_methods(self) -> tuple[int, int]:
        
        sum_ = sum(self.destroy_weights)
        destroy_probabilities = [x / sum_ for x in self.destroy_weights]
        d_method_idx = np.random.choice(range(len(self.destroy_methods)), p=destroy_probabilities)
        
        sum_ = sum(self.repair_weights)
        repair_probabilities = [x / sum_ for x in self.repair_weights]
        r_method_idx = np.random.choice(range(len(self.repair_methods)), p=repair_probabilities)
        
        return d_method_idx, r_method_idx
    
    ## Faz a atualização dos scores dos métodos
    def __update_scores(self, destroy_method_idx, repair_method_idx, sigma_flags) -> None:
        
        if sigma_flags[0]:
            self.destroy_scores[destroy_method_idx] += self.sigma_values[0]
            self.repair_scores[repair_method_idx] += self.sigma_values[0]
        elif sigma_flags[1]:
            self.destroy_scores[destroy_method_idx] += self.sigma_values[1]
            self.repair_scores[repair_method_idx] += self.sigma_values[1]
        elif sigma_flags[2]:
            self.destroy_scores[destroy_method_idx] += self.sigma_values[2]
            self.repair_scores[repair_method_idx] += self.sigma_values[2]
            
        return 
    
    ## Faz a atualização da quantidade de utilizações dos métodos
    def __update_times_used(self, destroy_method_idx, repair_method_idx) -> None:
        
        self.destroy_n_times_used[destroy_method_idx] += 1
        self.repair_n_times_used[repair_method_idx] += 1
        
        return 
    
    ## função que gera um nova solução 
    def __generate_new_solution(self, solution, destroy_method_idx, repair_method_idx):
        
        
        destroy_method = self.destroy_methods[destroy_method_idx]
        repair_method = self.repair_methods[repair_method_idx]
        
        return self.solution_generator(solution, destroy_method, repair_method, self.methods_hyperparameters)
    
    def run(self, seconds_limit: int = 300, max_iteration = np.inf, initial_temp: int = 100, verbose = False):
    
        x_solution = copy.deepcopy(self.initial_solution)
        x_cost = self.evaluation_function(x_solution)
        
        current_iter = 1 ## começa a iteração com valor 1 devido a função de atualização de temperatura
        current_temp = initial_temp
        min_temp = 1e-6 ## definindo uma temperatura mínima
        
        segment_size = 100 ## número de iterações necessárias para permitir atualizar os pesos dos métodos
        sigma_flags = [False, False, False]
        
        start_time = time.time()
        verbose_time = start_time
        
        while True:
            sigma_flags = [False, False, False] ## marca os casos de uso dos sigmas
            
            current_time = time.time()
            if(current_time - start_time) >= seconds_limit or current_iter > max_iteration: ## verificando condições de parada
                break
            
            d_method_idx, r_method_idx = self.__select_methods()
            xt_solution = self.__generate_new_solution(x_solution, d_method_idx, r_method_idx)
            xt_cost = self.evaluation_function(xt_solution)
            
            if xt_cost < self.best_cost: ### Atualização do melhor global
                
                self.time_to_best = time.time() - start_time ## atualiza o tempo que achou o melhor 
                sigma_flags[0] = True ## se a nova solução é a melhor global
                self.best_solution, self.best_cost = copy.deepcopy(xt_solution), xt_cost
           
            if xt_cost < x_cost: 
                sigma_flags[1] = True ## se a nova solução é melhor que a atual
            
            if current_temp >= min_temp: ## se a temperatura é menor que a mínima, ignora
                
                if (random.random() < math.exp( (x_cost - xt_cost) / current_temp)):
                    
                    sigma_flags[2] = True ## se a no solução é aceita
                    x_solution, x_cost = copy.deepcopy(xt_solution), xt_cost
                    
                current_temp = initial_temp / math.log(1 + current_iter)  
                
            ## faz printa de algumas informações em intervalos de 5 segundos caso o verbose esteja ligado
            if current_time - verbose_time >= 5 and verbose: 
                print(f'Iteration {current_iter}, Temperature {current_temp:.3f}, Best evaluation {self.best_cost:.5f}, Time {current_time - start_time}')
                print(f'Weights(d, r):: {self.destroy_weights} // {self.repair_weights}')
                print(f'Scores(d, r):: {self.destroy_scores} // {self.repair_scores}')
                
                verbose_time = current_time
            
            
            self.__update_scores(d_method_idx, r_method_idx, sigma_flags) ## atualização dos scores 
            self.__update_times_used(d_method_idx, r_method_idx) ## atualização do número de utilizações
            if current_iter % segment_size == 0: ## a cada segmento, atualizar os pesos
                self.__update_methods_weights()
                    
            self.costs_per_iter.append(x_cost) ## salvando custo da solução aceita da iteração atual
            current_iter +=1
            

    def get_results(self) -> dict:
        
        results = dict()
        results['eval_per_iter'] = self.costs_per_iter
        results['best_solution'] = self.best_solution
        results['best_cost'] = self.best_cost
        
        return results
    

## Classe que guarda as informações principais que serão utilizadas pelo programa
class CVRPInstance:
    
    def __init__(self, path: str):
            
        ## lendo com biblitoeca
        instance = vrplib.read_instance(path)
        
        ## salvando informações necessárias
        self.name = instance['name']
        self.number_trucks = int(instance['comment'].split(',')[1].split(':')[-1][1:]) ## pegando número de caminhões do comentário da instância
        self.number_nodes = instance['dimension']
        self.max_capacity = instance['capacity']
        self.node_coords = instance['node_coord']
        self.demands = instance['demand']
        self.depot_idx = int(instance['depot'][0])
                
        self.distance_matrix = self.__get_distance_matrix()
        self.optimal_value = int(instance['comment'].split(',')[-1].split(':')[-1][1:-1]) ## pegando valor ótimo do comentário da instância
        self.clients = [i for i in range(self.number_nodes) if i != self.depot_idx]
    
    def __get_distance_matrix(self):
        
        
        dist_matrix = [[0] * self.number_nodes for _ in range(self.number_nodes)]
           
        for i in range(self.number_nodes):
            for j in range(self.number_nodes):
                dx = self.node_coords[i][0] - self.node_coords[j][0]
                dy = self.node_coords[i][1] - self.node_coords[j][1]
                dist_matrix[i][j] = math.sqrt(dx*dx + dy*dy)
            
        return dist_matrix
    
    def __str__(self) -> str:
        return f'Name: {self.name}\nNumber of trucks: {self.number_trucks}\nNumber of nodes: {self.number_nodes}\nMax_capacity: {self.max_capacity}\nDepot_idx: {self.depot_idx}\nOptimal value: {self.optimal_value}'
