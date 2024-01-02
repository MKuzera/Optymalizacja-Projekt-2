import numpy as np
from scipy.optimize import differential_evolution

def funkcja_celu(x):
    return x[0]**2 + x[1]**2 - np.cos(2.5*np.pi*x[0]) - np.cos(2.5*np.pi*x[1]) + 2

bounds = [(-5, 5), (-5, 5)]
mutation_values = [0.01, 0.1, 1, 10, 100]

for mutation in mutation_values:
    print(f"Running optimizations for mutation value: {mutation}")
    for i in range(100):
        result = differential_evolution(funkcja_celu, bounds, mutation=mutation)
        print(f"Iteration {i+1}: Best individual: {result.x}, Best fitness: {result.fun}, Number of function calls: {result.nfev}")