import numpy as np

from HJClass import HJOptimizer
from HookeJevees import *

from RosenbrockClass import *
import csv




def funkcja_probna(x):

    return 2.5 * ((x[0] * x[0] - x[1])**2) + (1 - x[0])**2
def funkcjaCelu(x):
    x = np.array(x)

    return x[0]**2 + x[1]**2 - math.cos(2.5*math.pi*x[0]) - math.cos(2.5*math.pi*x[1]) + 2



if __name__ == '__main__':

    s0_values = [0.1, 0.01, 0.001]
    N = 100
    Nmax = 1000
    epsilon = 0.0001
    alfaHJ = 0.5 # w HJ alfa od 0 do 1
    alfaRB = 2.0 # w RB alfa > 1
    beta = 0.1



    randoms = np.array([np.random.uniform(-1, 1, 2) for _ in range(N)])

    with open('rb.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s0', 'x1(0)', 'x2(0)', 'x1*', 'x2*', 'y*', 'function_calls'])
        for s0 in s0_values:
            for i in range(N):
                x0 = randoms[i]
                rb = RosenbrockOptimizer(x0, np.array([s0, s0]), alfaRB, beta, epsilon, Nmax, funkcjaCelu)
                x, y, counter = rb.optimize()
                writer.writerow([s0, x0[0], x0[1], x[0], x[1], y, counter])


    with open('hj.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s', 'x1(0)', 'x2(0)', 'x1*', 'x2*', 'y*', 'function_calls'])
        for s in s0_values:
            for i in range(N):
                x0 = randoms[i]
                hj = HJOptimizer(x0, np.array([s, s]), alfaHJ, epsilon, Nmax, funkcjaCelu)
                x,y,counter =hj.optimize()
                writer.writerow([s, x0[0], x0[1], x[0], x[1], y, counter])

    # with open('hj.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #   #  writer.writerow(['s', 'x1(0)', 'x2(0)', 'x1*', 'x2*', 'y*', 'function_calls'])
    #     for s in s0_values:
    #         for i in range(N):
    #             x0 = randoms[i]
    #             x = hooke_jeeves(x0,s,epsilon,objective_function)
    #
    #
    #           #  hj = HJOptimizer(x0, np.array([s, s]), alfaHJ, epsilon, Nmax, funkcjaCelu)
    #          #  x,y,counter =hj.optimize()
    #
    #             x1 = objective_function(x)[0]
    #             print("p")
    #             print(x1[0])
    #             print(x1[1])
    #             print(funkcjaCelu([1 ,1]))
    #             print(funkcjaCelu([x1[0],x1[1]]))
    #             print(objective_function([x1[0], x1[1]]))
    #
    #             writer.writerow([s, x0[0], x0[1], x[0], funkcjaCelu([x1[0],x1[1]])])
    #
    #

    x = np.array([-0.5,1.0])
    s0 = np.array([0.5, 0.5])
    print("HJ")
    hj = HJOptimizer(x, s0, 0.1, 0.001, 10000, funkcja_probna)
    result = hj.optimize()
    print(result)
    print(funkcja_probna(result[0]))



    s0 = np.array([1.0, 1.0])
    rb = RosenbrockOptimizer(x, s0, 2.0, 0.5, 0.0001, 10000, funkcja_probna)
    result = rb.optimize()
    print("Rosenbrock")
    print(result)


