import numpy as np
from scipy.optimize import minimize

class RosenbrockOptimizer:
    def __init__(self, x0, s0, alfa, beta, epsilon, Nmax, fun, mode = None):
        self.x0 = x0
        self.s0 = s0
        self.alfa = alfa
        self.beta = beta
        self.epsilon = epsilon
        self.Nmax = Nmax
        self.fun = fun
        self.counter = 0
        self.mode = mode

    def norm(self, A):
        return np.linalg.norm(A)

    def funkcja(self, x):
        self.counter += 1
        return self.fun(x)

    def optimize(self):
        if self.mode:
            return  minimize(self.fun, self.s0, method='BFGS')
        n = len(self.x0)
        l = np.zeros(n)
        p = np.zeros(n)
        s = np.copy(self.s0)
        D = np.eye(n)
        x = np.copy(self.x0)

        while True:
            for i in range(n):
                xt = x + s * D[:, i]
                if self.funkcja(xt) < self.funkcja(x):
                    x = xt
                    l[i] += s
                    s *= self.alfa
                else:
                    p[i] += 1
                    s *= -self.beta

            change = any(p[i] != 0 and l[i] != 0 for i in range(n))

            if change:
                Q = np.outer(l, np.ones(n))
                Q *= D
                D = Q
                for i in range(1, n):
                    temp = np.dot(Q[:, i], D[:, :i])
                    D[:, i] = Q[:, i] - temp

                s = max(s * self.alfa, 1e-6)
                l = np.zeros(n)
                p = np.zeros(n)

            max_s = np.max(np.abs(s))

            if max_s < self.epsilon or self.counter > self.Nmax:
                return x, self.funkcja(x), self.counter