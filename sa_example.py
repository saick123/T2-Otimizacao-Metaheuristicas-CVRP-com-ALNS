
#https://www.geeksforgeeks.org/implement-simulated-annealing-in-python/

import math
import random

# Objective function: Rastrigin function
def objective_function(x):
    return 10 * len(x) + sum([(xi**2 - 10 * math.cos(2 * math.pi * xi)) for xi in x])

# Neighbor function: small random change
def get_neighbor(x, step_size=0.1):
    neighbor = x[:]
    index = random.randint(0, len(x) - 1)
    neighbor[index] += random.uniform(-step_size, step_size)
    return neighbor

# Simulated Annealing function
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # Initial solution
    best = [random.uniform(bound[0], bound[1]) for bound in bounds]
    best_eval = objective(best)
    current, current_eval = best, best_eval
    scores = [best_eval]

    for i in range(n_iterations):
        # Decrease temperature
        t = temp / float(i + 1)
        # Generate candidate solution
        candidate = get_neighbor(current, step_size)
        candidate_eval = objective(candidate)
        # Check if we should keep the new solution
        if candidate_eval < best_eval or random.random() < math.exp((current_eval - candidate_eval) / t):
            current, current_eval = candidate, candidate_eval
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
                scores.append(best_eval)

        # Optional: print progress
        if i % 100 == 0:
            print(f"Iteration {i}, Temperature {t:.3f}, Best Evaluation {best_eval:.5f}")

    return best, best_eval, scores

# Define problem domain
bounds = [(-5.0, 5.0) for _ in range(2)]  # for a 2-dimensional Rastrigin function
n_iterations = 1000
step_size = 0.1
temp = 10

# Perform the simulated annealing search
best, score, scores = simulated_annealing(objective_function, bounds, n_iterations, step_size, temp)

print(f'Best Solution: {best}')
print(f'Best Score: {score}')
