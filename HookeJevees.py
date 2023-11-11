
import math
import numpy as np

def funkcjaCelu(x):
    x = np.array(x)
    return x[0] * x[0]  + x[1]*x[1]  - math.cos(2.5*math.pi*x[0]) - math.cos(2.5*math.pi*x[1]) + 2




# Hooke'a-Jeevesa.
# Dane wejściowe:
# punkt startowy x
# długość kroku s
# współczynnik zmniejszania długości kroku 0 < alfa < 1
# dokładność epsilon > 0
# maksymalna liczba wywołań funkcji celu Nmax

# funkcja szuka minimum w siatce AxB
# x0 - ma byc [x,x]
# s - ma byc [s,s] (czyli jak podane 0.5 podajesz[0.5 , 0.5])
# Poprawic jakos zeby jeszcze elegancko ilosc wywolan funkcji wyswietlalo/zwracalo
# HJ_pronuj - no to uzywamy wew funkcji

def HJ(x0,s,alfa,epsilon,Nmax,funkcja):

    xb = np.copy(x0)
    xb_old = np.copy(x0)
    print("xb {} x {} xbold {}".format(xb, x0, xb_old))
    counter =0
    while(True):
        counter+=1
        print("XD")
        print("xb {} x {} xbold {}".format(xb,x0,xb_old))
        x = HJ_probuj(xb,s,funkcja)
        if funkcja(x) < funkcja(xb):
            print("funkcja(x) {} < funkcja(xb) {}".format(funkcja(x) , funkcja(xb)))
            while(True):
                xb_old = np.copy(xb)
                xb = np.copy(x)
                x = 2.0*xb - xb_old
                x = HJ_probuj(x,s,funkcja)
                if funkcja(x) >= funkcja(xb):
                    break
                if counter > Nmax:
                    print("Counter max")
                    return xb
        else:
            s = s*alfa
        if np.linalg.norm(s) < epsilon or counter > Nmax:
            return xb

def HJ_probuj(xb,s,funkcja):

    n = xb.size
    D = np.array([[1,0],[0,1]])
    for i in range(n):
        D[i][i] = 1


    x = np.array([0,0])

    for i in range(n):
        x = xb + s*D[i]
        print("X {}".format(x))
        if funkcja(x) < funkcja(xb):
            xb = np.copy(x)
        else:
            x = xb - s*D[i]
            if funkcja(x) < funkcja(xb):
                xb = np.copy(x)
    return xb




