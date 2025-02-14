import os
import vrplib
import math
import random
import time
import copy

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



def initial_solution_generator(instance : CVRPInstance):

    routes = []
    clients = []
    max_capacity = instance.max_capacity

    while True: 
        routes = []
        clients = [c for c in range(instance.number_nodes) if c != instance.depot_idx]
    
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

def get_load_from_route(route):
    total = 0
    for node_idx in route:
        total += instance.demands[node_idx]
    return total

            
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
                
#check_all_problems()
##instance = CVRPInstance(path=teste_path)
##init_sol = initial_solution_generator(instance)
##print(init_sol)

def evaluation_function(solution: list[list[int]]):
   
    total_cost = 0
    for route in solution:
        for i in range(len(route) - 1):
            total_cost += instance.distance_matrix[route[i]][route[i+1]]
        total_cost += instance.distance_matrix[instance.depot_idx][route[0]]
        total_cost += instance.distance_matrix[instance.depot_idx][route[-1]]
        
    return total_cost


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
   
    candidate_solution = None
   
    while True:    
        parcial, unassigned = random_removal(copy.deepcopy(solution), int(instance.number_nodes*0.15))
        candidate_solution = greedy_repair(parcial, unassigned)
        if candidate_solution is None:
            continue 
        break
    return  candidate_solution

############ fazer um método de destruir e outro de reparar pra testar
class ALNS:
    
    def __init__(self, feasible_solution: list[list[int]], evaluation_foo):
        
        self.evaluation_function = evaluation_foo
        self.p_iminus = []
        self.p_plus = []
        self.destroy_methods = []
        self.repair_methods = []
        self.initial_solution = copy.deepcopy(feasible_solution)
        self.best_solution = copy.deepcopy(feasible_solution)
        self.best_cost = self.evaluation_function(self.best_solution) 
        
    def run(self, seconds_limit: int = 300, alpha: int = 0.999, intial_temp: int = 100):
    
        x_solution = [x for x in self.best_solution]
        x_cost = self.evaluation_function(x_solution)
        
        start_time = time.time()
        while True:
            
            current_time = time.time()
            if(current_time - start_time) >= 300:
                break
            
            
            
                
        


if __name__ == '__main__':

    teste_path = '/home/matheusmartin/Área de Trabalho/T2-Otimizacao-Metaheuristicas-CVRP-com-ALNS/Vrp-Set-A/A/A-n32-k5.vrp'
   
    instance = CVRPInstance(path=teste_path)
    init_sol = initial_solution_generator(instance)
    while True:
        
        parcial, unassigned = random_removal(copy.deepcopy(init_sol), int(instance.number_nodes*0.2))
        candidate_solution = greedy_repair(parcial, unassigned)
        
        if candidate_solution is None:
            continue
                        
        print(evaluation_function(init_sol), evaluation_function(candidate_solution))
        
        break

    
    
    