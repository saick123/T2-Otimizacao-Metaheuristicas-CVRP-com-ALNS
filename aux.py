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
        for i in range(len(route) - 1):
            total_cost += dist[route[i]][route[i+1]]
        total_cost += dist[instance.depot_idx][route[0]]
        total_cost += dist[instance.depot_idx][route[-1]]
        
    return int(total_cost)


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
            return None
        else:
            parcial_solution[route_idx].insert(insertion_idx, client)
        
    return parcial_solution


def verify_solution(solution: list[list[int]]):

    for route in solution:
        if get_load_from_route(route) > instance.max_capacity:
            return False
    
    client_counter = {c : 0 for c in instance.clients}
    for route in solution:
        for c in route:
            client_counter[c] +=1
        
    for c in instance.clients:
        if client_counter[c] !=1 :
            return False
    
    return True

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

############ fazer um método de destruir e outro de reparar pra testar
class ALNS:
    
    def __init__(self, feasible_solution: list[list[int]], evaluation_foo, destroy_methods: list = [], repair_methos: list = []):
        
        self.evaluation_function = evaluation_foo
        self.p_iminus = []
        self.p_plus = []
        self.destroy_methods = []
        self.repair_methods = []
        self.initial_solution = copy.deepcopy(feasible_solution)
        self.best_solution = copy.deepcopy(feasible_solution)
        self.best_cost = self.evaluation_function(self.best_solution) 
    
    ################## 
    ### as fuções vão receber a solução(cópia) e os hiperparâmetros que é um dicionáro dado pelo usuário
    def __gen_new_sol(self, solution):
        pass
        
    def run(self, seconds_limit: int = 300, max_iteration = np.inf, alpha: int = 0.999, initial_temp: int = 100, verbose = False):
    
        x_solution = copy.deepcopy(self.initial_solution)
        x_cost = self.evaluation_function(x_solution)
        
        current_iter = 1 ##### começar na iteração 1
        current_temp = initial_temp
        min_temp = 1e-6
        costs = []
        
        start_time = time.time()
        while True:
            
            current_time = time.time()
            
            if(current_time - start_time) >= seconds_limit or current_iter > max_iteration:
                break
            
            xt_solution = generate_new_solution(x_solution)
            xt_cost = evaluation_function(xt_solution)
            
            flag_new_best = False ### usado para não fazer cópias desnecessárias
            if xt_cost < self.best_cost:
                
                flag_new_best = True
                x_solution, x_cost = copy.deepcopy(xt_solution), xt_cost
                self.best_solution, self.best_cost = copy.deepcopy(xt_solution), xt_cost
                costs.append(self.best_cost)
            
            if current_temp > min_temp:
                
                if (random.random() < math.exp( (x_cost - xt_cost) / current_temp)) and not flag_new_best:
                    x_solution, x_cost = copy.deepcopy(xt_solution), xt_cost
                    
                current_temp = initial_temp / math.log(1 + current_iter)
                #current_temp = initial_temp / (current_iter +  1)
                #current_temp = current_temp * alpha    
            
            if current_iter % 1000 == 0 and verbose: ## desativei rapidinho
                print(f'Iteration {current_iter}, Temperature {current_temp:.3f}, Best evaluation {self.best_cost:.5f}, Time {current_time - start_time}')
            
            
            current_iter +=1
            


def run_all():
    
    global instance, files

    for case_name, filepaths in files:
        for vrp_path, sol_path in filepaths:
            instance = CVRPInstance(path=vrp_path)
            init_solution = initial_solution_generator(instance)
            alns = ALNS(init_solution, evaluation_function)
            alns.run(initial_temp=10000, max_iteration=10000, verbose=False)
            print(f'FOLDER {case_name}, VRP_PATH = {vrp_path}, Best eval {alns.best_cost}, Well-Know Best cost {instance.optimal_value}, Feasible? {verify_solution(alns.best_solution)}')


if __name__ == '__main__':

    run_all()
        