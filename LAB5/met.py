import csv
import math

import numpy as np


num_function_calls = 0
class Solution:
    def __init__(self, x1, x2, mutation_rate):
        global num_function_calls
        self.x1 = x1
        self.x2 = x2
        self.mutation_rate = mutation_rate
        self.fit = self.fit_fun()
        num_function_calls += 1

    def fit_fun(self):
        return self.x1**2 + self.x2**2 - math.cos(2.5*math.pi*self.x1)-math.cos(2.5*math.pi*self.x2) + 2

def EA(N, epsilon, Nmax, O):
    global num_function_calls
    mi = 20
    lambda_ = 40
    P = [Solution(np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.01) for _ in range(mi + lambda_)]
    Pm = [Solution(0, 0, 0) for _ in range(mi)]

    for _ in range(Nmax):
        for i in range(mi + lambda_):
            P[i].fit = P[i].fit_fun()
            num_function_calls += 1
            if P[i].fit < epsilon:
             #   print(f"{P[i].fit} eps{epsilon} fcalls{num_function_calls} ")
                return P[i], num_function_calls

        IFF = [1 / P[i].fit for i in range(mi)]
        s_IFF = sum(IFF)

        for i in range(lambda_):
            r = np.random.uniform(0, s_IFF)
            s = 0
            for j in range(mi):
                s += IFF[j]
                if s >= r:
                    P[mi + i] = P[j]
                    break

        for i in range(mi, mi + lambda_):
            P[i].x1 += P[i].mutation_rate * np.random.normal()
            P[i].x2 += P[i].mutation_rate * np.random.normal()

        for i in range(mi, mi + lambda_, 2):
            P[i].x1, P[i+1].x1 = P[i+1].x1, P[i].x1
            P[i].x2, P[i+1].x2 = P[i+1].x2, P[i].x2

        for i in range(mi, mi + lambda_):
            P[i].fit = P[i].fit_fun()
            num_function_calls += 1
            if P[i].fit < epsilon:
                return P[i], num_function_calls

        P.sort(key=lambda x: x.fit)
        Pm = P[:mi]

        if Pm[0].fit < epsilon:
            return Pm[0], num_function_calls

    return Pm[0], num_function_calls

#best_solution, num_function_calls = EA(10, 0.001, 10000, None)
#print(f"x1: {best_solution.x1}, x2: {best_solution.x2}, fit: {best_solution.fit} calls:  {num_function_calls}")

#
# jesli epsilon 0.001 to nic nie moze znalezc
# jak 0.01 to cos znajduje
# 0.1 znajduje ale bardzo nie dokladne
# przy nmax 3000 jedna iteracja sie liczy 1s xD
# i tak zle wyniki sa wiekszosc
#


mutation_values = [0.01, 0.1, 1, 10, 100]
dane = []

for mutation in mutation_values:
    print(f"Running optimizations for mutation value: {mutation}")
    for i in range(100):
        best_solution, num_function_calls = EA(mutation, 0.01,1000, None)
        print(f"Iter {i} x1: {best_solution.x1}, x2: {best_solution.x2}, fit: {best_solution.fit} calls:  {num_function_calls}")
        dane.append([i + 1, best_solution.x1, best_solution.x2,best_solution.fit, num_function_calls])
        num_function_calls =0


csv_filename = "iterations_data.csv"

csv_headers = ["Iteration", "x1", "x2", "Best Fitness", "Number of Function Calls"]
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_headers)
    csv_writer.writerows(dane)

print(f"Dane zapisane do pliku CSV: {csv_filename}")