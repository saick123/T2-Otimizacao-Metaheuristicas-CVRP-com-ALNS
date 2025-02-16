import os
import vrplib
import math
import random
import time
import copy
import numpy as np

folders = ['Vrp-Set-' + c for c in ['A', 'B', 'F']]
files = []
instance = None


for folder in folders:
    innter_folder_path = os.path.join(folder, folder[-1])
    files_path = sorted(list({os.path.join(innter_folder_path, s[:-4]) for s in os.listdir(innter_folder_path)}))
    files_ = ( folder, [(fp + '.vrp', fp + '.sol') for fp in files_path])
    files.append(files_)

class CVRPInstance:
    
    def __init__(self, path: str):
        
        """ CHAVES DA INSTÂNCIA
        name
        comment
        type
        dimension
        edge_weight_type
        capacity
        node_coord
        demand
        depot
        edge_weight
        """
        
        instance = vrplib.read_instance(path)
        
        self.name = instance['name']
        self.number_trucks = int(instance['comment'].split(',')[1].split(':')[-1][1:])
        self.number_nodes = instance['dimension']
        self.max_capacity = instance['capacity']
        self.node_coords = instance['node_coord']
        self.demands = instance['demand']
        self.depot_idx = int(instance['depot'][0])
        self.distance_matrix = instance['edge_weight']
        self.optimal_value = int(instance['comment'].split(',')[-1].split(':')[-1][1:-1])
        self.clients = [i for i in range(self.number_nodes) if i != self.depot_idx]
        
    def __str__(self) -> str:
        
        return f'Name: {self.name}\nNumber of trucks: {self.number_trucks}\nNumber of nodes: {self.number_nodes}\nMax_capacity: {self.max_capacity}\nDepot_idx: {self.depot_idx}\nOptimal value: {self.optimal_value}'


            
if False:
    print(teste['name'])
    N = teste['dimension']
    coords = teste['node_coord']
    m = [[0]*N]*N

    dist = teste['edge_weight']
    print(coords)

    row = []
    for j in range(len(m)):
        dx = (coords[0][0] - coords[j][0])
        dy = (coords[0][1] - coords[j][1])
        d = math.sqrt(dx*dx + dy*dy)
        row.append(d)
    print(dist[0])
    print(row)


def get_distance(node1 : int , node2 : int):
    dist = instance.distance_matrix
    return int(dist[node1][node2])
    
def initial_solution_generator(instance : CVRPInstance) -> list[list[int]]:

    routes = []
    clients = []
    max_capacity = instance.max_capacity

    while True: 
        routes = []
        clients = copy.deepcopy(instance.clients)
    
        for _ in range(instance.number_trucks):
            truck_load = 0
            route = []
            
            while clients:
                c_idx = random.choice(clients)
                clients.remove(c_idx)
                demand = instance.demands[c_idx]
                if demand + truck_load <= max_capacity:
                    route.append(c_idx)
                    truck_load += demand
                else:
                    clients.append(c_idx)
                    break
                
            routes.append(route)
        
        if len(clients) == 0: break
        
    return routes

###### mudei 
def get_load_from_route(route):
    demands = instance.demands
    total = 0
    for node_idx in route:
        total += demands[node_idx]
    return total

### lixo       
def check_all_problems():
    i = 1
    for name, problems in files:
        for vrp, sol in problems:
            instance = CVRPInstance(path=vrp)
            print(i, vrp)
            initial_solution = initial_solution_generator(instance)
            print('solução inicial gerada')
            if i == 4 and False:
                exit(1)
            i+=1    
                

def evaluation_function(solution: list[list[int]]):
    dist = instance.distance_matrix
    total_cost = 0
    for route in solution:
        if len(route) == 0: continue
        
        for i in range(len(route) - 1):
            total_cost += dist[route[i]][route[i+1]]
        total_cost += dist[instance.depot_idx][route[0]]
        total_cost += dist[instance.depot_idx][route[-1]]
        
    return math.ceil(total_cost)


def random_removal(solution: list[list[int]], q: int):
    
    ## criando cópia
    clients = [x for x in instance.clients]
    D = []
    q_ = q
    
    while q_:
        r_idx = random.randint(0, instance.number_trucks - 1)
        if len(solution[r_idx]) == 0: continue ## rota está vazia, tentar pegar outra
        c_idx = random.randint(0, len(solution[r_idx]) - 1)
        D.append(solution[r_idx].pop(c_idx))
        q_ -=1

    return solution, D

def removal_cost(route, client, client_idx):
    dist = instance.distance_matrix
    succ = instance.depot_idx if client_idx == len(route) - 1 else route[client_idx + 1]
    pred = instance.depot_idx if client_idx == 0 else route[client_idx - 1]
    return dist[pred][client] + dist[client][succ] - dist[pred][succ]

def worst_removal(solution: list[list[int]], q: int):
    
    r_costs = []
    D = []
    
    for route_idx, route in enumerate(solution):
        for c_idx, c in enumerate(route):
            r_cost = removal_cost(route, c, c_idx)
            r_costs.append([c, route_idx, r_cost])

    r_costs.sort(key=lambda x: x[-1], reverse=True)
    q_ = q
    p = 3 # verificar quais são valores de p possíveis aqui
    
    while q_:
        y = random.random()
        idx = math.floor(len(r_costs)*(math.pow(y, p)))
        
        client, route_idx, r_cost = r_costs.pop(idx)
        
        D.append(client)
        solution[route_idx].remove(client)
        q_ -=1
    
    return solution, D

def worst_removal_consertado(solution: list[list[int]], q: int):
    
    r_costs = []
    D = []
    q_ = q
    p = 3 # verificar quais são valores de p possíveis aqui
    
    while q_:
        
        r_costs = []
        for route_idx, route in enumerate(solution):
            for c_idx, c in enumerate(route):
                r_cost = removal_cost(route, c, c_idx)
                r_costs.append([c_idx, route_idx, r_cost])

        r_costs.sort(key=lambda x: x[-1], reverse=True)
        
        y = random.random()
        idx = math.floor(len(r_costs)*(math.pow(y, p)))
        
        c_idx, route_idx, r_cost = r_costs.pop(idx)
        
        c = solution[route_idx].pop(c_idx)
        D.append(c)
        
        q_ -=1
    
    return solution, D

## tá sem o componente de mesma rota
def compute_relatedness(client_1, client_2, alpha=0.75, beta=0.1):
    
    dist = instance.distance_matrix
    demands = instance.demands
    
    r_ij = alpha * dist[client_1][client_2] + beta * abs(demands[client_1] - demands[client_2])
    return r_ij
    
def shaw_removal(solution: list[list[int]], q: int):
    
    D = []
    p = 3
    
    non_empty_routes = [i for i in range(len(solution)) if len(solution[i]) != 0]
    r_idx = random.choice(non_empty_routes)
    c_idx = random.randint(0, len(solution[r_idx]) - 1)
    D.append(solution[r_idx].pop(c_idx))
    
    while len(D) < q:
        
        ref_client = random.choice(D)
        relatedness_list = []
        
        for route_idx, route in enumerate(solution):
            for client_idx, client in enumerate(route):
                r_value = compute_relatedness(ref_client, client)
                relatedness_list.append((r_value, client_idx, route_idx))
                
        relatedness_list.sort(key= lambda x: x[0])
        
        y = random.random()
        idx = math.floor(len(relatedness_list)*(math.pow(y, p)))
        
        r_value, client_idx, route_idx = relatedness_list[idx]
        removed_client = solution[route_idx].pop(client_idx)        
        
        D.append(removed_client)
        
    return solution, D


def valid_insertion(client_idx, route):
    
    if get_load_from_route(route) + instance.demands[client_idx] > instance.max_capacity:
        return False
    return True


def isertion_cost(route, client_idx, insertion_idx):

    succ = instance.depot_idx if insertion_idx == len(route) else route[insertion_idx]
    pred = instance.depot_idx if insertion_idx == 0 else route[insertion_idx - 1]
    return instance.distance_matrix[pred][client_idx] + instance.distance_matrix[client_idx][succ] - instance.distance_matrix[pred][succ]
    

def best_insertion(client_idx, parcial_solution):
    
    best_cost, best_route_idx, best_idx = None, None, None
    
    for route_idx, route in enumerate(parcial_solution):
                
        if not valid_insertion(client_idx, route):
            continue

        for i in range(0, len(route) + 1):
            
            cost = isertion_cost(route, client_idx, i)
            if best_cost is None or cost < best_cost:
                best_cost, best_route_idx, best_idx = cost, route_idx, i 
                
        
    return best_route_idx, best_idx 
    
def greedy_repair(parcial_solution:list[list[int]], D: list[int]):
    
    while len(D) != 0:
        client = D.pop(0)
        route_idx, insertion_idx = best_insertion(client, parcial_solution) 
        
        if route_idx is None or insertion_idx is None:
            return parcial_solution
        else:
            parcial_solution[route_idx].insert(insertion_idx, client)
        
    return parcial_solution


def random_repair(parcial_solution: list[list[int]], D: list[int]):
    
    
    while (len(D) != 0):
        
        c_idx = random.choice(range(len(D)))
        client = D.pop(c_idx)
        
        possible_insertion_routes = [i for i in range(len(parcial_solution)) if valid_insertion(client, parcial_solution[i])]
        if len(possible_insertion_routes) == 0:
            return parcial_solution ### vai quebrar no generate
        
        r_idx = random.choice(possible_insertion_routes)
        insertion_idx = random.randint(0, len(parcial_solution[r_idx]))
        parcial_solution[r_idx].insert(insertion_idx, client)
    
    return parcial_solution
        
            
def verify_solution(solution: list[list[int]]):

    for route in solution:
        if get_load_from_route(route) > instance.max_capacity:
            return 1
    
    client_counter = {c : 0 for c in instance.clients}
    for route in solution:
        for c in route:
            client_counter[c] +=1
        
    for c in instance.clients:
        if client_counter[c] !=1 :
            return 2
    
    return 0

def generate_new_solution(solution: list[list[int]]):
    q = int(instance.number_nodes*0.1)
    candidate_solution = None
    while True:    
        #parcial, unassigned = random_removal(copy.deepcopy(solution), q)
        parcial, unassigned = worst_removal(copy.deepcopy(solution), q)
        candidate_solution = greedy_repair(parcial, unassigned)
        if candidate_solution is None:
            continue 
        break
    return  candidate_solution

def real_best_insertion(client_idx, parcial_solution, routes_idx):
    
    best_cost, best_route_idx, best_idx = None, None, None
    
    for r_idx in routes_idx:
    
        route = parcial_solution[r_idx]
                
        #if not valid_insertion(client_idx, route):
            #continue
        for i in range(0, len(route) + 1):
            cost = isertion_cost(route, client_idx, i)
            if best_cost is None or cost < best_cost:
                best_cost, best_route_idx, best_idx = cost, r_idx, i 
                
        
    return best_cost, best_route_idx, best_idx 

def real_greedy_repair(parcial_solution : list[list[int]], D: list[int]):
    
    
    best_positions = [(np.inf, -1, -1) for _ in D]
    routes_idx = list(range(len(parcial_solution)))
    #index_list = list(range(len(parcial_solution))) ## auxiliar pra função de max
    N = len(D)
    
    for _ in range(N):
        
        for idx, c in enumerate(D):
            best_cost, route_idx, insertion_idx = real_best_insertion(c, parcial_solution, routes_idx)
            if best_cost < best_positions[idx][0]:
                best_positions[idx] = (best_cost, route_idx, insertion_idx)
            #best_positions.append((best_cost, route_idx, insertion_idx))
    
        best_idx = min(range(len(best_positions)), key=lambda k : best_positions[k][0])
        
        best_client = D.pop(best_idx)
        best_cost, best_route_idx, best_insertion_idx = best_positions.pop(best_idx)
        
        parcial_solution[best_route_idx].insert(best_insertion_idx, best_client)
        routes_idx = [best_route_idx]
        
    return parcial_solution


def best_insertion_in_route(client, route):
    
    best_cost, best_idx = None, None
    
    for i in range(0, len(route) + 1):
        
        cost = isertion_cost(route, client, i)
        if best_cost is None or cost < best_cost:
            best_cost, best_idx = cost, i 
            
    return best_idx
    

def secure_greedy_repair(parcial_solution: list[list[int]], D: list[int]):
    
    
    def special_valid_insertion(parcial_solution_, client_, route_idx_, lazy_insertion_clients_):
        demands = instance.demands
        max_capacity = instance.max_capacity
        total_load = get_load_from_route(parcial_solution_[route_idx_]) + sum([demands[c] for c in lazy_insertion_clients_[route_idx_]])
        total_load += demands[client_]
        
        if total_load > max_capacity : return  False
        return True
        
    lazy_insertion_clients = [[] for _ in range(len(parcial_solution))] ## clientes lazy pra cada rota
        
    while (len(D) != 0):
        
        c_idx = random.choice(range(len(D)))
        client = D.pop(c_idx)
        
        possible_insertion_routes = [i for i in range(len(parcial_solution)) if special_valid_insertion(parcial_solution, client, i, lazy_insertion_clients)]
        if len(possible_insertion_routes) == 0:
            return parcial_solution ### vai quebrar no generate
        
        route_idx = random.choice(possible_insertion_routes)
        lazy_insertion_clients[route_idx].append(client)
        
        #insertion_idx = best_insertion_in_route(client, parcial_solution[route_idx])
        #parcial_solution[route_idx].insert(insertion_idx, client)
        
    
    #print(lazy_insertion_clients)
    
    for lazy_idx, lazy_clients in enumerate(lazy_insertion_clients):
        if len(lazy_clients) == 0: continue
        
        while len(lazy_clients):
            
            best_positions = [(np.inf, -1) for _ in lazy_clients]
            
            for client_idx, client in enumerate(lazy_clients):
                best_cost, _, best_insertion_idx =  real_best_insertion(client, parcial_solution, [lazy_idx])
                best_positions[client_idx] = (best_cost, best_insertion_idx)
            
            best_lazy_client_idx = min(range(len(best_positions)), key= lambda k: best_positions[k][0])
            
            best_client = lazy_clients.pop(best_lazy_client_idx) ## tirando o cliente escolhido
            _, best_insertion_idx = best_positions[best_lazy_client_idx]
            
            parcial_solution[lazy_idx].insert(best_insertion_idx, best_client)
        
    return parcial_solution


############ fazer um método de destruir e outro de reparar pra testar
class ALNS:
    
    def __init__(self, feasible_solution: None, evaluation_foo = None, 
                 destroy_methods: list = None, repair_methods: list = None, 
                 r: int = 0.1, sigma_values: list[int] = [], q_interval: tuple[int, int] = None):
        
        assert(q_interval is not None)
        assert(destroy_methods is not None)
        assert(repair_methods is not None)
        assert(evaluation_foo is not None)
        assert(feasible_solution is not None)
        
            
        self.evaluation_function = evaluation_foo
        self.initial_solution = copy.deepcopy(feasible_solution)
        self.best_solution = copy.deepcopy(feasible_solution)
        self.best_cost = self.evaluation_function(self.best_solution) 
        
        self.sigma_values = sigma_values
        self.r = r
        self.q_interval = q_interval
        
        self.destroy_methods = destroy_methods
        self.repair_methods = repair_methods
        self.__init_destroy_repair_methods_metadata()
    
    ################## 
    ### as fuções vão receber a solução(cópia) e os hiperparâmetros que é um dicionáro dado pelo usuário
    def __init_destroy_repair_methods_metadata(self):
        # weights ok, scores ok, times_used ok,
        
        self.repair_weights = [1]*(len(self.repair_methods)) ## pesos dos métodos são inicializados com tudo 1
        self.destroy_weights = [1]*(len(self.destroy_methods))
        self.repair_scores = [0]*(len(self.repair_methods))
        self.destroy_scores = [0]*(len(self.destroy_methods))
        self.repair_n_times_used = [0]*(len(self.repair_methods))
        self.destroy_n_times_used = [0]*(len(self.destroy_methods))
        
    
    def __update_methods_weights(self) -> None:
        
        for i in range(len(self.destroy_methods)):
            if self.destroy_n_times_used[i] > 0:
                self.destroy_weights[i] = self.destroy_weights[i]*(1 - self.r) + self.r * (self.destroy_scores[i] / self.destroy_n_times_used[i])
            self.destroy_n_times_used[i] = 0
            self.destroy_scores[i] = 0

        for i in range(len(self.repair_methods)):
            if self.repair_n_times_used[i] > 0:
                self.repair_weights[i] = self.repair_weights[i]*(1 - self.r) + self.r * (self.repair_scores[i] / self.repair_n_times_used[i])
            self.repair_n_times_used[i] = 0
            self.repair_scores[i] = 0
            
        return 
    
    def __select_methods(self) -> tuple[int, int]:
        
        sum_ = sum(self.destroy_weights)
        destroy_probabilities = [x / sum_ for x in self.destroy_weights]
        d_method_idx = np.random.choice(range(len(self.destroy_methods)), p=destroy_probabilities)
        
        sum_ = sum(self.repair_weights)
        repair_probabilities = [x / sum_ for x in self.repair_weights]
        r_method_idx = np.random.choice(range(len(self.repair_methods)), p=repair_probabilities)
        
        return d_method_idx, r_method_idx
    
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
    
    def __update_times_used(self, destroy_method_idx, repair_method_idx) -> None:
        
        self.destroy_n_times_used[destroy_method_idx] += 1
        self.repair_n_times_used[repair_method_idx] += 1
        
        return 
    
    def __get_next_temp(self, initial_temp, lambda_, time_elapsed):
        return initial_temp * math.exp(-(lambda_*time_elapsed))
    
    def __generate_new_solution(self, solution, destroy_method_idx, repair_method_idx):
        
        q = random.randint(self.q_interval[0], self.q_interval[1])
        candidate_solution = None
        tries = 0
        
        while True: ## fica tentando novamente até encontrar uma solução viável
            
            q = random.randint(self.q_interval[0], self.q_interval[1])
            tries +=1
            solution_copy = copy.deepcopy(solution)
            #parcial, unassigned = random_removal(copy.deepcopy(solution), q)
            
            parcial_solution, D = self.destroy_methods[destroy_method_idx](solution_copy, q)
            candidate_solution = self.repair_methods[repair_method_idx](parcial_solution, D)
            
            code = verify_solution(candidate_solution)
            if code == 0:
                break
            
            if tries >= 1000000:
                raise('Numero de tentativas de construir solução ultrapassou 1.000.000')
        
        return  candidate_solution
        
    
    def run(self, seconds_limit: int = 300, max_iteration = np.inf, initial_temp: int = 100, verbose = False):
    
        x_solution = copy.deepcopy(self.initial_solution)
        x_cost = self.evaluation_function(x_solution)
        
        current_iter = 1 ##### começa na iteração número 1 devido ao cáculo da temperatura (T_0/log(1 + iter))
        current_temp = initial_temp
        min_temp = 1e-6
        costs = []
        segment_size = 100 ## número de iterações necessárias para permitir atualizar os pesos dos métodos
        sigma_flags = [False, False, False]
        #lambda_ = (math.log( initial_temp / min_temp ) / seconds_limit)
        
        start_time = time.time()
        verbose_time = start_time
        
        while True:
            sigma_flags = [False, False, False] ## reiniciar a cada iteração
            
            current_time = time.time()
            if(current_time - start_time) >= seconds_limit or current_iter > max_iteration:
                break
            
            
            d_method_idx, r_method_idx = self.__select_methods()
            xt_solution = self.__generate_new_solution(x_solution, d_method_idx, r_method_idx)
            xt_cost = self.evaluation_function(xt_solution)
            
            #xt_solution = generate_new_solution(x_solution)
            #xt_cost = evaluation_function(xt_solution)
            
            #flag_new_best = False ### usado para não fazer cópias desnecessárias
            if xt_cost < self.best_cost: ### ATUALIZANDO O MELHOR GLOBAL
                
                sigma_flags[0] = True ## se a nova solução é a melhor global
                #flag_new_best = True
                #x_solution, x_cost = copy.deepcopy(xt_solution), xt_cost
                self.best_solution, self.best_cost = copy.deepcopy(xt_solution), xt_cost
                costs.append(self.best_cost)
           
            if xt_cost < x_cost: 
                sigma_flags[1] = True ## se a nova solução é melhor que a atual
            
            if current_temp >= min_temp:
                
                if (random.random() < math.exp( (x_cost - xt_cost) / current_temp)):
                    
                    sigma_flags[2] = True ## se a no solução é aceita
                    x_solution, x_cost = copy.deepcopy(xt_solution), xt_cost
                    
                current_temp = initial_temp / math.log(1 + current_iter)
                #current_temp = self.__get_next_temp(initial_temp, lambda_, current_time - start_time)
                #current_temp = initial_temp / (current_iter +  1)
                #current_temp = current_temp * alpha    
            
            if current_time - verbose_time >= 5 and verbose: 
                print(f'Iteration {current_iter}, Temperature {current_temp:.3f}, Best evaluation {self.best_cost:.5f}, Time {current_time - start_time}')
                print(f'Weights(d, r):: {self.destroy_weights} // {self.repair_weights}')
                print(f'Scores(d, r):: {self.destroy_scores} // {self.repair_scores}')
                
                verbose_time = current_time
            
            
            self.__update_scores(d_method_idx, r_method_idx, sigma_flags)
            self.__update_times_used(d_method_idx, r_method_idx)
            if current_iter % segment_size == 0: ## a cada segmento, atualizar os pesos
                self.__update_methods_weights()
                    
            
            current_iter +=1
            


def run_all():
    
    global instance, files

    for case_name, filepaths in files:
        if case_name != 'Vrp-Set-F' : continue
        print(f'CASE NAME : {case_name}')
        for vrp_path, sol_path in filepaths:
            instance = CVRPInstance(path=vrp_path)
            init_sol = initial_solution_generator(instance)
            
            destroy_methods = [random_removal, worst_removal_consertado, shaw_removal]
            repair_methods = [greedy_repair, random_repair]
            q_min = int(0.05*len(instance.clients))
            q_max = int(0.30*len(instance.clients))
            q_interval = (q_min, q_max)
            
            alns = ALNS(feasible_solution=init_sol, evaluation_foo=evaluation_function,
                    destroy_methods= destroy_methods, repair_methods= repair_methods,
                    r=0.1, sigma_values=[3,2,1],
                    q_interval=q_interval)
        
            #init_temp = evaluation_function(init_sol) * (-0.05/math.log(0.5))
            #print(init_temp)
            #exit()
            alns.run(seconds_limit=300, initial_temp=1000, verbose=True)
        
            #alns = ALNS(init_solution, evaluation_function)
            #alns.run(initial_temp=10000, max_iteration=10000, verbose=False)
            print(f'OK {vrp_path}')
            with open('log.txt', 'a') as out_file:
                print('\n------------------------------- INFORMATION -------------------------------', file=out_file)
                print(f'FOLDER {case_name}, VRP_PATH = {vrp_path}, Best eval {alns.best_cost}, Well-Know Best cost {instance.optimal_value}, Feasible? {verify_solution(alns.best_solution)}', file=out_file)
                print(f'Best Solution: {alns.best_solution}', file=out_file)
                print()


if __name__ == '__main__':


    run_all()

    if False:
        
        test_path = '/mnt/c/Users/mathe/OneDrive/Área de Trabalho/teste/T2-Otimizacao-Metaheuristicas-CVRP-com-ALNS/Vrp-Set-A/A/A-n32-k5.vrp'
        instance = CVRPInstance(path=test_path)
        init_sol = initial_solution_generator(instance)
        q_min = int(0.10*len(instance.clients))
        q_max = int(0.30*len(instance.clients))
        
        parcial, D = shaw_removal(init_sol, q_min)
        print('Parcial', parcial)
        print('D', D)
        new_sol = secure_greedy_repair(parcial, D)
        
        print(verify_solution(new_sol), evaluation_function(new_sol), new_sol)
        print(instance.optimal_value, init_sol)
        
        if False:
            destroy_methods = [shaw_removal]
            repair_methods = [greedy_repair]
            
            alns = ALNS(feasible_solution=init_sol, evaluation_foo=evaluation_function,
                        destroy_methods= destroy_methods, repair_methods= repair_methods,
                        r=0.1, sigma_values=[3,2,1],
                        q_interval=(q_min, q_max))
            
            alns.run(seconds_limit=30, initial_temp=10000, verbose=True)
            print(verify_solution(alns.best_solution), alns.best_cost, instance.optimal_value)
            #run_all()
                