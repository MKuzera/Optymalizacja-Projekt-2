import numpy as np


class HJOptimizer:
    def __init__(self, x0, s, alfa, epsilon, Nmax, fun):
        self.x0 = x0
        self.s0 = s
        self.alfa = alfa
        self.epsilon = epsilon
        self.Nmax = Nmax
        self.fun = fun
        self.counter = 0

    def norm(self, A):
        N = np.sum(A ** 2)
        return np.sqrt(N)

    def funkcja(self, x):
        self.counter += 1
        return self.fun(x)

    def HJ_probuj(self, xb, s):
        n = xb.size
        D = np.eye(n)

        x = np.zeros(n)
        for i in range(n):
            x = xb + s[i] * D[i]

            if self.funkcja(x) < self.funkcja(xb):
                xb = np.copy(x)
            else:
                x = xb - s[i] * D[i]
                if self.funkcja(x) < self.funkcja(xb):
                    xb = np.copy(x)

        return xb

    def optimize(self):
        xb = np.copy(self.x0)
        xb_old = np.copy(self.x0)
        s = np.copy(self.s0)

        while True:
            self.counter += 1
            x = self.HJ_probuj(xb, s)

            if self.funkcja(x) < self.funkcja(xb):
                while True:
                    xb_old = np.copy(xb)
                    xb = np.copy(x)
                    x = 2.0 * xb - xb_old
                    x = self.HJ_probuj(x, s)
                    if self.funkcja(x) >= self.funkcja(xb):
                        break
                    if self.counter > self.Nmax:
                        print("Counter max")
                        return xb,self.funkcja(xb), self.counter-1
            else:
                s = s * self.alfa

            if np.linalg.norm(s) < self.epsilon or self.counter > self.Nmax:
                return xb,self.funkcja(xb), self.counter-1