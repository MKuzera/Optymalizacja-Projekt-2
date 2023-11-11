from HookeJevees import *
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = np.array([-0.5,1])
    s0 = np.array([0.5, 0.5])
    wynik = HJ(x, s0, 0.5, 0.00001, 1000, funkcja_probna)
    print(wynik)
    print(funkcja_probna(wynik))
    #print(hooke_jeeves(funkcja_probna,xnp,0.5,0.0001))
