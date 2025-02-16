import heapq

def compute_all_insertions(client, solution, distances, demands, capacity):
    """ Calcula todas as possíveis inserções de um cliente em todas as rotas. """
    insertion_options = []

    for vehicle, route in solution.items():
        if sum(demands[c] for c in route) + demands[client] > capacity:
            continue  # Se não há capacidade suficiente, pule essa rota

        # Testa todas as posições possíveis na rota
        for i in range(len(route) + 1):
            new_route = route[:i] + [client] + route[i:]
            cost = sum(distances[new_route[j]][new_route[j+1]] for j in range(len(new_route) - 1))
            insertion_options.append((cost, vehicle, i))  # (Custo, Veículo, Posição)

    return client, sorted(insertion_options)  # Ordena por custo crescente


def regret_k_insertion(solution, unallocated_clients, distances, demands, capacity, k=2):
    """ Implementa a heurística Regret-k considerando múltiplas inserções por rota. """
    
    best_insertions = {}  # Armazena as melhores inserções de cada cliente

    # Inicializa as melhores inserções
    for client in unallocated_clients:
        _, insertion_options = compute_all_insertions(client, solution, distances, demands, capacity)
        if insertion_options:
            best_insertions[client] = insertion_options  # Salva todas as opções ordenadas

    while unallocated_clients:
        regret_values = []

        # Calcula o regret para cada cliente
        for client, insertion_options in best_insertions.items():
            if len(insertion_options) >= k:
                best_cost = insertion_options[0][0]
                regret_value = insertion_options[k-1][0] - best_cost  # Regret-k
            else:
                regret_value = float("inf")  # Se há menos de k opções, força a inserção logo

            heapq.heappush(regret_values, (-regret_value, client, insertion_options[0]))

        # Escolhe o cliente com maior regret e insere na melhor posição
        _, best_client, (best_cost, best_vehicle, best_position) = heapq.heappop(regret_values)
        solution[best_vehicle].insert(best_position, best_client)
        unallocated_clients.remove(best_client)

        # Atualiza apenas clientes afetados
        affected_clients = [c for c in unallocated_clients if best_vehicle in [v for _, v, _ in best_insertions[c]]]
        
        for client in affected_clients:
            _, new_options = compute_all_insertions(client, solution, distances, demands, capacity)
            best_insertions[client] = new_options

    return solution
