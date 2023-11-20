import numpy as np

class RosenbrockOptimizer:
    def __init__(self, x0, s0, alfa, beta, epsilon, Nmax, fun):
        self.x0 = x0
        self.s0 = s0
        self.alfa = alfa
        self.beta = beta
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

    def optimize(self):
        n = self.x0.size
        l = np.zeros(n)
        p = np.zeros(n)
        s = np.copy(self.s0)
        D = np.eye(n)
        x = np.copy(self.x0)
        y = self.funkcja(x)

        while True:
            for i in range(n):
                xt = x + s[i] * D[i]
                if self.funkcja(xt) < self.funkcja(x):
                    x = xt
                    l[i] += s[i]
                    s[i] *= self.alfa
                 #   y = self.funkcja(x)
                else:
                    p[i] += 1
                    s[i] *= -self.beta

            change = all(p[i] != 0 and l[i] != 0 for i in range(n))

            if change:
                Q = np.outer(l, np.ones(n))
                Q = D * Q
                V = Q[0] / self.norm(Q[0])
                D[:, 0] = V
                for i in range(1, n):
                    temp = np.zeros(n)
                    for j in range(i):
                        temp += np.dot(np.transpose(Q[i]), D[j]) * D[j]
                    V = Q[i] - temp
                    D[:, i] = V

                s = self.s0
                l = np.zeros(n)
                p = np.zeros(n)

            max_s = np.max(np.abs(s))


            if max_s < self.epsilon or self.counter > self.Nmax:
                return x, self.funkcja(x), self.counter
