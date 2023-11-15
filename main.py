from HJClass import HJOptimizer
from HookeJevees import *
from Rosenbrock import *
from RosenbrockClass import *
import csv
#cel ->
# wprowdzic dane
# zrobic te 2 funkcje
# wprowadzci ta funkcje testowa  isprawdzic na niej algorytmy
# przygotowac funkcje problemu rzeczywistego..
# wykresy i inne
# sprawko
# noi fajrancik



def funkcja_probna(x):

    return 2.5 * ((x[0] * x[0] - x[1])**2) + (1 - x[0])**2
def funkcjaCelu(x):
    x = np.array(x)

    return x[0]**2 + x[1]**2 - math.cos(2.5*math.pi*x[0]) - math.cos(2.5*math.pi*x[1]) + 2e-3



if __name__ == '__main__':
    s0_values = [0.5, 0.05, 0.005]
    N = 100
    Nmax = 1000
    epsilon = 0.000000000001
    alfa = 2
    beta = 0.5

    with open('rb.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s0', 'x1(0)', 'x2(0)', 'x1*', 'x2*', 'y*', 'function_calls'])
        for s0 in s0_values:
            for i in range(N):
                x0 = np.random.uniform(-5, 5, 2)
                rb = RosenbrockOptimizer(x0, np.array([s0, s0]), alfa, beta, epsilon, Nmax, funkcjaCelu)
                x, y, counter = rb.optimize()
                writer.writerow([s0, x0[0], x0[1], x[0], x[1], y, counter])
    with open('hj.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s', 'x1(0)', 'x2(0)', 'x1*', 'x2*', 'y*', 'function_calls'])
        for s in s0_values:
            for i in range(N):
                x0 = np.random.uniform(-5, 5, 2)
                hj = HJOptimizer(x0, np.array([s, s]), alfa, epsilon, Nmax, funkcjaCelu)
                x,y,counter =hj.optimize()
                writer.writerow([s, x0[0], x0[1], x[0], x[1], y, counter])

    x = np.array([-0.5,1])
    s0 = np.array([0.5, 0.5])
    print("HJ")
    hj = HJOptimizer(x, s0, 0.5, 0.00001, 1000, funkcja_probna)
    result = hj.optimize()
    print(result)
    print(funkcja_probna(result[0]))



    s0 = np.array([1.0, 1.0])
    rb = RosenbrockOptimizer(x, s0, 2.0, 0.5, 0.0001, 10000, funkcja_probna)
    result = rb.optimize()
    print("Rosenbrock")
    print(result)


