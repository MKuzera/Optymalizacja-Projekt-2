import random
import numpy as np
from deap import base, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Global variable to track the number of function calls
num_function_calls = 0

def funkcja_celu(individual):
    global num_function_calls
    num_function_calls += 1
    return individual[0]**2 + individual[1]**2 - np.cos(2.5*np.pi*individual[0]) - np.cos(2.5*np.pi*individual[1]) + 2,

toolbox.register("evaluate", funkcja_celu)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

mutation_values = [0.01, 0.1, 1, 10, 100]

for mutation in mutation_values:
    print(f"Running optimizations for mutation value: {mutation}")
    for i in range(100):
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=mutation, indpb=0.1)
        toolbox.register("select", tools.selBest)

        pop = toolbox.population(n=100)
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        for g in range(NGEN):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

        best_ind = tools.selBest(pop, 1)[0]
        print(f"Iteration {i+1}: Best individual: {best_ind}, Best fitness: {best_ind.fitness.values}, Number of function calls: {num_function_calls}")
        num_function_calls = 0  # Reset the counter for the next optimization