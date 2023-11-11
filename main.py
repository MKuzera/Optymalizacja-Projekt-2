from HookeJevees import *
from Rosenbrock import *
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
    return x[0] * x[0]  + x[1]*x[1]  - math.cos(2.5*math.pi*x[0]) - math.cos(2.5*math.pi*x[1]) + 2



if __name__ == '__main__':
    x = np.array([-0.5,1])
    s0 = np.array([0.5, 0.5])
    print("HJ")
    wynik = HJ(x, s0, 0.5, 0.00001, 1000, funkcja_probna)
    print(wynik)
    print(funkcja_probna(wynik))
    s0 = np.array([1.0, 1.0])
    wynik = Rosenbrock(x, s0, 2.0, 0.5, 0.001, 1000, funkcja_probna)
    print("Rosenbrock")
    print(wynik)
    print(funkcja_probna(wynik))

