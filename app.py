import os
import vrplib
import math
import random
import time
import copy
import numpy as np
import sys
import matplotlib.pyplot as plt

from lib import *


## instancia é uma variável global para faciliar a passagem de informações entre os operadores
instance = None


## gera uma solução inicial para a instância    
def initial_solution_generator(instance : CVRPInstance) -> list[list[int]]:

    routes = []
    clients = []
    max_capacity = instance.max_capacity

    #Fica em loop até conseguir criar uma solução válida
    while True: 
        routes = []
        clients = copy.deepcopy(instance.clients)
    
        for _ in range(instance.number_trucks): ##Tenta completar um caminhão por vez
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
        
        if len(clients) == 0: break ## se todos os caminhões foram preenchidos e não restaram clientes, pode sair do loop
                
    return routes

## Calcula o total de carga dentro da rota
def get_load_from_route(route):
    demands = instance.demands
    total = 0
    for node_idx in route: ## soma a demanda de todos os clientes dentro da rota
        total += demands[node_idx]
    return total

## Calcula o custo de uma solução passada     
def evaluation_function(solution: list[list[int]]):
    dist = instance.distance_matrix
    total_cost = 0
    for route in solution:
        if len(route) == 0: continue ## se a rota não possui clientes, pula a rota
        
        for i in range(len(route) - 1):
            total_cost += dist[route[i]][route[i+1]] ## somando distância entre clientes
        total_cost += dist[instance.depot_idx][route[0]] ## somando distância do depósito para o primeiro cliente
        total_cost += dist[instance.depot_idx][route[-1]] ## somando distância do último cliente para o depósito
        
    return total_cost


## Remove aleatoriamente q clientes da solução passada
def random_removal(solution: list[list[int]], parameters: dict):
    
    q = parameters['q']
    
    D = []
    q_ = q
    
    while q_:
        r_idx = random.randint(0, instance.number_trucks - 1) ## sorteia uma rota para tirar um cliente
        if len(solution[r_idx]) == 0: continue ## rota está vazia, tentar pegar outra
        
        c_idx = random.randint(0, len(solution[r_idx]) - 1) ## sorteia um cliente dentro da rota
        D.append(solution[r_idx].pop(c_idx)) ## remove o cliente
        q_ -=1

    return solution, D ## retorna a solução parcial e a lista D de clientes removidos


## Calcula o custo de remoção de um cliente, ou seja, o custo da solução com ele menos o custo da solução sem o cliente
def removal_cost(route, client, client_idx):
    dist = instance.distance_matrix
    succ = instance.depot_idx if client_idx == len(route) - 1 else route[client_idx + 1] ##se ele for o último, o seu sucessor é o depósito
    pred = instance.depot_idx if client_idx == 0 else route[client_idx - 1] ## se ele for o primeiro, seu antecessor é o depósito
    return dist[pred][client] + dist[client][succ] - dist[pred][succ] ## calcula o custo


## Remove q clientes da solução passada, dando preferência para aqueles com maior custo de remoção
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
    
    return solution, D ## retorna a solução parcial e a lista D de clientes removidos


## Remove q clientes da solução passada, dando preferência para aqueles com maior custo de remoção
def worst_removal_consertado(solution: list[list[int]], parameters: dict):
    
    q = parameters['q']
    p = parameters['p']
    
    r_costs = [] ## vetor com tuplas (client_idx, route_idx, removal_cost), guarda os valore de remoção de todos os clientes
    D = []
    q_ = q
    
    
    while q_:
        
        r_costs = []
        for route_idx, route in enumerate(solution):
            for c_idx, c in enumerate(route):
                r_cost = removal_cost(route, c, c_idx) ## Calcula custo da remoção
                r_costs.append([c_idx, route_idx, r_cost]) ## Coloca no vetor

        r_costs.sort(key=lambda x: x[-1], reverse=True) ## Ordena do maior para o menor
        
        y = random.random()
        idx = math.floor(len(r_costs)*(math.pow(y, p))) ## Seleciona um elemento da ponta aleatório
        
        c_idx, route_idx, r_cost = r_costs.pop(idx)
        
        c = solution[route_idx].pop(c_idx) ## retira da solução e coloca em D
        D.append(c)
        
        q_ -=1
    
    return solution, D ## retorna a solução parcial e a lista D de clientes removidos


## Calcula o nível de similaridade entre dois clientes
def compute_relatedness(client_1, client_2, alpha=0.75, beta=0.1):
    
    dist = instance.distance_matrix
    demands = instance.demands
    
    r_ij = alpha * dist[client_1][client_2] + beta * abs(demands[client_1] - demands[client_2]) ## distância entre os clientes somada com a diferença no valor das demandas
    return r_ij


## Remove q clientes da solução considerando o nível de similaridade entre eles
def shaw_removal(solution: list[list[int]], parameters: dict):
    
    q = parameters['q']
    p = parameters['p']
    alpha = parameters['alpha']
    beta = parameters['beta']
    
    D = []
    
    non_empty_routes = [i for i in range(len(solution)) if len(solution[i]) != 0] ## rotas não vazias
    r_idx = random.choice(non_empty_routes)
    c_idx = random.randint(0, len(solution[r_idx]) - 1) ## seleciona um cliente aleatório para iniciar no conjunto D
    D.append(solution[r_idx].pop(c_idx))
    
    while len(D) < q:
        
        ref_client = random.choice(D) ## Pega um elemento aleatório em D, chamado cliente referência
        relatedness_list = [] ## Salva vários tipos de informação -> (relatedness_value, client_idx, route_idx)
        
        for route_idx, route in enumerate(solution):
            for client_idx, client in enumerate(route):
                r_value = compute_relatedness(ref_client, client, alpha=alpha, beta=beta) ## Calcula o valor do nível de similaridade entre o cliente referência e os clientes restantes na solução
                relatedness_list.append((r_value, client_idx, route_idx)) 
                
        relatedness_list.sort(key= lambda x: x[0]) ## quando menor o valor de r_value, mais relacionados estão, então ordenar do menor para o maior
        
        y = random.random()
        idx = math.floor(len(relatedness_list)*(math.pow(y, p))) ## Seleciona um elemento da ponta aleatório
        
        r_value, client_idx, route_idx = relatedness_list[idx]
        removed_client = solution[route_idx].pop(client_idx) ## Remove o cliente da solução e insere no conjunto D       
        
        D.append(removed_client)
        
    return solution, D ## retorna a solução parcial e a lista D de clientes removidos


## Verifica se uma inserção é válida
def valid_insertion(client, route):
    
    if get_load_from_route(route) + instance.demands[client] > instance.max_capacity: ## Verifica se a soma ultrapassa a capacidade
        return False
    return True


## Calcula o custo de inserção de um cliente em uma rota em uma determinada posição
def isertion_cost(route, client, insertion_idx):

    succ = instance.depot_idx if insertion_idx == len(route) else route[insertion_idx] ## Se ele é o último, seu sucessor é o depósito
    pred = instance.depot_idx if insertion_idx == 0 else route[insertion_idx - 1] ## Se ele é o primeiro, seu antecessor é o depósito
    return instance.distance_matrix[pred][client] + instance.distance_matrix[client][succ] - instance.distance_matrix[pred][succ]
    
    
## Retorna dentre a melhor posição de inserção possível para aquele cliente
def best_insertion(client, parcial_solution):
    
    best_cost, best_route_idx, best_idx = None, None, None
    
    for route_idx, route in enumerate(parcial_solution):
                
        if not valid_insertion(client, route): ## se estourar a capacidade, não tenta inserir
            continue

        for i in range(0, len(route) + 1): ## analisa todas as posições possíveis
            
            cost = isertion_cost(route, client, i)
            if best_cost is None or cost < best_cost:
                best_cost, best_route_idx, best_idx = cost, route_idx, i 
                
    return best_route_idx, best_idx ## retorna o índice da melhor rota e o índice da melhor inserção na rota


## Repara a solução ao inserir clientes de forma gulosa na solução parcial
def greedy_repair(parcial_solution:list[list[int]], D: list[int]):
    
    while len(D) != 0: ## itera sobre os clientes
        client = D.pop(0)
        route_idx, insertion_idx = best_insertion(client, parcial_solution)  ## pegando melhor inserção do cliente
        
        if route_idx is None or insertion_idx is None: ## se não foi possível inserir em nenhuma posição, retorna a solução incompleta (significa que a reparação falhou)
            return parcial_solution
        else:
            parcial_solution[route_idx].insert(insertion_idx, client) ## senão insere
        
    return parcial_solution


## Repara a solução de forma aleatória
def random_repair(parcial_solution: list[list[int]], D: list[int]):
    
    
    while (len(D) != 0): ## Enquanto houve clientes para inserir
        
        c_idx = random.choice(range(len(D)))
        client = D.pop(c_idx)
        
        possible_insertion_routes = [i for i in range(len(parcial_solution)) if valid_insertion(client, parcial_solution[i])]
        if len(possible_insertion_routes) == 0: ## se não existe nenhuma rota capaz de o cliente ser inserido, retorna a solução incompleta (significa que a reparação falhou)
            return parcial_solution 
        
        r_idx = random.choice(possible_insertion_routes)
        insertion_idx = random.randint(0, len(parcial_solution[r_idx]))
        parcial_solution[r_idx].insert(insertion_idx, client) ## inserindo cliente em uma das possíveis rotas válidas
    
    return parcial_solution

## Verifica se uma solução é válida ou não
def verify_solution(solution: list[list[int]]):

    for route in solution:
        if get_load_from_route(route) > instance.max_capacity: ## Se alguma rota ultrapassa a capacidade máxima, retorna o código de erro 1
            return 1
    
    client_counter = {c : 0 for c in instance.clients}
    for route in solution:
        for c in route:
            client_counter[c] +=1
        
    for c in instance.clients:
        if client_counter[c] !=1 : ## Se algum cliente foi encontrado mais de uma vez ou nenhuma vez, retorna o código de erro 2
            return 2
    ## retorna 0 se deu tudo certo
    return 0 


##########################################################################################################################################################
def real_best_insertion(client_idx, parcial_solution, routes_idx):
    
    best_cost, best_route_idx, best_idx = None, None, None
    
    for r_idx in routes_idx:
    
        route = parcial_solution[r_idx]
                
        if not valid_insertion(client_idx, route):
            continue
        
        for i in range(0, len(route) + 1):
            cost = isertion_cost(route, client_idx, i)
            if best_cost is None or cost < best_cost:
                best_cost, best_route_idx, best_idx = cost, r_idx, i 
                
        
    return best_cost, best_route_idx, best_idx 
##########################################################################################################################################################

##########################################################################################################################################################
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
##########################################################################################################################################################


## Calcula todas as inserções possíveis  para um determinado cliente somente em rotas específicas
def compute_all_insertions(client, parcial_solution, routes_idx):
    
    insertion_positions = [] ## salva (custo, índice da rota, índice de inserção na rota)
    
    for route_idx in routes_idx:
        
        if not valid_insertion(client, parcial_solution[route_idx]): ## se a inserção não é válida, não inserir
            continue
        
        for insertion_idx in range(len(parcial_solution[route_idx]) + 1):
            cost = isertion_cost(parcial_solution[route_idx], client, insertion_idx)
            insertion_positions.append((cost, route_idx, insertion_idx)) ## salvando posição

    return insertion_positions
    
## Repara a solução inserindo os clientes considerando o arrependimento ("regret") das melhores K posições 
def regret_k(partial_solution: list[list[int]], D: list[int], k = 2):
    
    
    insertion_positions_dict = {c : [] for c in D} ## Dicionário que salva todas as inserções possíveis para um cliente
    
    routes_idx = list(range(len(partial_solution)))
    for client_idx, client in enumerate(D):
        insertion_positions = compute_all_insertions(client, partial_solution, routes_idx) ## calculando inserções possíveis
        insertion_positions.sort(key= lambda x : x[0])  ## ordena as posições do menor custo de inserção para o maior
        insertion_positions_dict[client] = insertion_positions
    
    while len(D): ## enquanto tiver cliente para inserir
        
        regret_values = [] ## salva (valor de arrependimento, cliente)
        
        for client_idx, client in enumerate(D):
            insertion_positions = insertion_positions_dict[client]
            if len(insertion_positions) < k and len(insertion_positions) != 0: ## se o client tem menos de k posições para ser inserido
                regret_values.append((np.inf, client)) ## ele ganha prioridade máxima
                
            elif len(insertion_positions) == 0: ## se não foi possível inserir o cliente
                return partial_solution ## retorna a solução de forma incompleta
            else:
                best_cost = insertion_positions[0][0]
                regret_value = 0
                for j in range(1, k): ## calculando o regret value
                    regret_value += (insertion_positions[j][0] - best_cost)
                regret_values.append((regret_value, client))
        
        regret_values.sort(key= lambda x: x[0], reverse=True) ## ordena do maior para o menor
        
        _, best_client = regret_values[0]
        insertion_positions = insertion_positions_dict[best_client]
        _, best_route_idx, best_insertion_idx = insertion_positions[0]
        
        partial_solution[best_route_idx].insert(best_insertion_idx, best_client) ## seleciona o cliente com maior valor de "regret" e insere
        D.remove(best_client)
        insertion_positions_dict.pop(best_client)
        
        for client_idx, client in enumerate(D):
            insertion_positions = insertion_positions_dict[client]
            to_update_flag = False
            for _, route_idx, _ in insertion_positions: ## verifica se o cliente precisa ser atualizado, ou seja, alguma de suas posições de inserção é na mesma rota que foi inserido o cliente anterior
                if route_idx == best_route_idx:
                    to_update_flag = True
                    break
            
            if to_update_flag: ## remove as posições da rota afetada e recalcula somente para a rota em questão
                new_insertion_positions_on_best_route = compute_all_insertions(client, partial_solution, [best_route_idx])
                insertion_positions = [x for x in insertion_positions if x[1] != best_route_idx]
                insertion_positions = insertion_positions + new_insertion_positions_on_best_route
                insertion_positions.sort(key= lambda x : x[0])             
                insertion_positions_dict[client] = insertion_positions
    
    return partial_solution

def regret_2(partial_solution: list[list[int]], D: list[int]): ## wrapper para o regret-2
    return regret_k(partial_solution, D, k=2)

def regret_3(partial_solution: list[list[int]], D: list[int]): ## wrapper para o regret-3
    return regret_k(partial_solution, D, k=3)


def new_solution_generator(solution, destroy_method, repair_method, hyperparameters: dict):
    
    q_interval = hyperparameters['q_interval']
    q = random.randint(q_interval[0], q_interval[1])
    candidate_solution = None
    tries = 0
    
    while True: ## fica tentando novamente até encontrar uma solução viável
        
        q = random.randint(q_interval[0], q_interval[1])
        tries +=1
        solution_copy = copy.deepcopy(solution)
        
        hyperparameters['q'] = q
        
        parcial_solution, D = destroy_method(solution_copy, hyperparameters)
        candidate_solution = repair_method(parcial_solution, D)
        
        code = verify_solution(candidate_solution)
        if code == 0: ## Encontrou uma solução viável
            break
        
        if tries >= 1000000:
            raise('Numero de tentativas de construir solução ultrapassou 1.000.000')
    
    return  candidate_solution



## Gera um gráfico onde no eixo X tem as iterações e no eixo Y o valor da função objetivo na iteração
## Além disso coloca uma linha horizontal sobre o Y de melhor resultado da função objetivo
def create_image(save_path, instance_name, image_name, best_cost, costs_per_iter):
  
    iterations = list(range(len(costs_per_iter)))
    max_x = len(costs_per_iter) - 1
    min_y = min(costs_per_iter)
    min_y_i = costs_per_iter.index(min_y)
    
    plt.figure(figsize=[12,6])
    plt.title(f"{instance_name} - Gráfico de Evolução da Função de Custo")
    plt.plot(iterations, costs_per_iter, linestyle = "dashdot", color=(102 / 255, 153 / 255, 255 / 255))
    plt.axhline(y = best_cost, color = 'g', linestyle = '--') 
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.scatter([min_y_i], [min_y], color='g', zorder=3)
    plt.text(min_y_i, min_y - 50, f"({min_y_i}, {min_y})", fontsize=8, verticalalignment='bottom', horizontalalignment="center", color="blue")
    plt.savefig(os.path.join(save_path, image_name + '.svg'))
    plt.close()

## Faz o parse dos argumentos de linha de comando, argumentos tem a forma "arg=arg_value"
def parse_args() -> dict:    
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in sys.argv[1:]}
    return args

def get_args() -> dict:
    
    args_dict = parse_args()
    
    ## Convertendo os argumentos de linha
    args_dict['q_max'] = float(args_dict['q_max'])
    args_dict['r'] = float(args_dict['r'])    
    args_dict['save_path'] = str(args_dict['save_path'])
    args_dict['number_iterations'] = int(args_dict['number_iterations'])
    args_dict['targets'] = list(args_dict['targets'].split(','))
    args_dict['time'] = int(args_dict['time'])
    args_dict['verbose'] = True if int(args_dict['verbose']) == 1 else False
    args_dict['figure'] = True if int(args_dict['figure']) == 1 else False
    
    return args_dict

def get_args_2()-> dict:
    
    args_dict = parse_args()
    args_dict['config_path'] = str(args_dict['config_path'])
    
    return args_dict

def read_config() -> dict:
    config_path = get_args_2()['config_path']
    config = dict()
    
    def convert_to_type(value : str , type_ : str ):
        match type_:
            case 'int':
                value = int(value)
            case 'float':
                value = float(value)
            case 'str':
                value = str(value)
        return value
               
    with open(config_path, 'r') as in_file:
        for line in in_file:
            if line[0] == '#' : continue
            splt = line.split(':')
            var = splt[0].strip()
            splt = splt[-1].split('=')
            type_ = splt[0].strip()
            value = splt[-1].strip()
            config[var] = convert_to_type(value, type_)
    return config

## salva todos os tipos de resultados requeridos nas tabelas do trabalho em um arquivo
## A primeira linha é o header, a segunda as informções e depois temos as rotas da melhor solução gerada
def save_results(save_path, q_max, r, times_to_best, best_costs, best_solution, optimal_solution=None):
    
    f_sol_optimal = instance.optimal_value
    if optimal_solution is not None:
        f_sol_optimal = evaluation_function(optimal_solution)
    
    f_mh_min = min(best_costs)
    f_mh_mean = sum(best_costs) / len(best_costs)
    instance_name = instance.name
    temp_min = min(times_to_best)
    temp_mean = sum(times_to_best) / len(times_to_best)
    gap_min = (f_mh_min - f_sol_optimal) / f_sol_optimal
    gap_mean = (f_mh_mean - f_sol_optimal) / f_sol_optimal
    save_path = os.path.join(save_path, f'{instance_name}-q_max-r-{q_max}-{r}.txt')
    
    header = 'instance; q_max; r; f_sol_optimal; f_mh_min; f_mh_mean; temp_min; temp_mean; gap_min; gap_mean'

    with open(save_path, 'w') as out_file:
        print(header, file=out_file)
        print(f'{instance_name}; {q_max}; {r}; {f_sol_optimal}; {f_mh_min}; {f_mh_mean}; {temp_min}; {temp_mean}; {gap_min}; {gap_mean}', file=out_file)
        print(file=out_file)
        print(f'Best Solution({f_mh_min}):',file=out_file)
        for k in range(len(best_solution)):
            print(f'route {k+1} ==>',best_solution[k], file=out_file)
        print(f"IsValid?: {verify_solution(best_solution)}", file=out_file)
        print(file=out_file)
        if optimal_solution is not None:
            print(f'Optimal Solution({f_sol_optimal}):',file=out_file)
            for k in range(len(optimal_solution)):
                print(f'route {k+1} ==>',optimal_solution[k], file=out_file)            
    

def read_solution(path: str):
    
    solution = []
    with open(path) as in_file:
        for line in in_file:
            if line[0] != "R":
                continue
            splt = line.split(':')[-1].strip()
            splt = [int(x.strip()) for x in splt.split(' ')]
            solution.append(splt)
    return solution

def main():
    
    global instance
    
    args = get_args()    
    
    destroy_methods = [random_removal, worst_removal_consertado, shaw_removal]
    repair_methods = [greedy_repair, regret_2, regret_3]
    
    ## Pegando hiperparâmetros
    q_max = args['q_max']
    q_min = 0.05
    r = args['r']
    
    hyperparameters = dict()
    hyperparameters['p'] = 3
    hyperparameters['alpha'] = 0.75
    hyperparameters['beta'] = 0.1
    
    sigma_values = [1, 0.4, 0.25]
    
    for vrp_path in args['targets']: ## para cada instância alvo
        
        instance = CVRPInstance(path=vrp_path)
        optimal_solution = read_solution(vrp_path.replace('.vrp', '.sol'))
        
        instance_size = len(instance.clients)
        
        q_interval_min = int(q_min * instance_size)
        q_interval_max = int(q_max * instance_size)
        q_interval = (q_interval_min, q_interval_max)
        
        times_to_best = []
        best_costs = []
        
        hyperparameters['q_interval'] = q_interval
        
        for w in range(args['number_iterations']): ## rodando diversas iterações 
             
            init_sol = initial_solution_generator(instance)
        
            alns = ALNS(feasible_solution=init_sol, evaluation_foo=evaluation_function,
                    destroy_methods= destroy_methods, repair_methods= repair_methods,
                    r=r, sigma_values=sigma_values, 
                    methods_hyperparameters=hyperparameters, solution_generator=new_solution_generator)
        
            alns.run(seconds_limit=args['time'], initial_temp=1000, verbose=args['verbose'])
            best_costs.append(alns.best_cost)
            times_to_best.append(alns.time_to_best)
            
            if args['figure']:
                create_image(args['save_path'], instance.name, f'{instance.name}-q_max-r-iter-{q_max}-{r}-{w+1}', alns.best_cost, alns.costs_per_iter) ## para cada iteração, salva o gráfico       
            
        save_results(args['save_path'], q_max, r, times_to_best, best_costs, alns.best_solution, optimal_solution=optimal_solution) ## salvando resultados finais

    return 

if __name__ == '__main__':

    main()