import numpy as np
# Dane wejściowe:
# punkt startowy x(0)
# wektor długość kroków s(0)
# współczynnik ekspansji α > 1,
# współczynnik kontrakcji 0 < β < 1
# dokładność ε > 0
# maksymalna liczba wywołań funkcji celu Nmax

def norm(A):
    N = np.sum(A**2)
    return np.sqrt(N)

def Rosenbrock(x0,s0,alfa,beta,epsilon,Nmax,funkcja):
    counter = 0
    n = x0.size
    l = np.zeros(n)
    p = np.zeros(n)
    s = np.copy(s0)
    D = np.zeros((n,n))
    for i in range(n):
        D[i][i] = 1
    x = np.copy(x0)
    y = funkcja(x)
    while(True):
        counter+=1
        for i in range(n):
            xt = x + s[i]*D[i]
            if funkcja(xt) < y:
                x = xt
                l[i] += s[i]
                s[i] *= alfa
                y = funkcja(x)
            else:
                p[i] = p[i] + 1
                s[i] = s[i] * (-beta)
        change = True
        for i in range(n):
            if p[i] ==0 or l[i] == 0:
                change = False
                break
        if change:
            Q = np.zeros((n,n))
            V = np.zeros(n)
            for i in range(n):
                for j in range(i+1):
                    Q[i][j] = l[i]
            Q = D*Q
            V = Q[0] / norm(Q[0])
            D[:, 0] = V # set_col(D,v,0)
            i = 0
            j =0
            while(i < n):
                temp = np.zeros(n)
                while j<i:
                    transpose = np.transpose(Q[i])
                    temp = temp + (transpose*D[j]) * D[j]
                    j+=1
                V = Q[i] - temp
                D[:, i] = V
                i+=1
            s = s0
            l = np.zeros(n)
            p = np.zeros(n)
        max_s = abs(s[0])
        for i in range(n):
            if max_s < abs(s[i]):
                max_s = abs(s[i])
        if(max_s < epsilon or counter > Nmax):
            return x, y, counter