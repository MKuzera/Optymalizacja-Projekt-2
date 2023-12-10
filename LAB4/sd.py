import numpy as np
import math
class Solution:
    f_calls = 0
    g_calls = 0
    h_calls = 0
    def __init__(self, x):
        self.x = x
        self.grad()
        self.hess()

    def grad(self):
        self.g_calls += 1
        self.g = np.zeros(2)
        self.g[0] = 10 * self.x[0] + 8 * self.x[1] - 34
        self.g[1] = 8 * self.x[0] + 10 * self.x[1] - 38

    def hess(self):
        self.h_calls += 1
        self.h= np.zeros((2,2))
        self.h[0][0] = 10
        self.h[0][1] = 0
        self.h[1][0] = 0
        self.h[1][1] = 10

    def obj_fun(self):
        self.f_calls += 1
        self.y= (self.x[0] + 2 * self.x[1] - 7)**2 + (2 * self.x[0] + self.x[1] - 5)**2
def compute_b(x, d, limits):
    n = x.shape[0]
    b = 1e9
    for i in range(n):
        if d[i] == 0:
            bi = 1e9
        elif d[i] > 0:
            bi = (limits[i, 1] - x[i]) / d[i]
        else:
            bi = (limits[i, 0] - x[i]) / d[i]
        if b != bi:
            b = bi
    return b


def golden(a, b, epsilon, Nmax, O):
    alfa = (math.sqrt(5) - 1) / 2
    A = Solution(a)
    B = Solution(b)
    C = Solution(B.x - alfa * (B.x - A.x))
    C.obj_fun(O)
    D = Solution(A.x + alfa * (B.x - A.x))
    D.obj_fun(O)
    while True:
        if C.y < D.y:
            B, D = D, C
            C.x = B.x - alfa * (B.x - A.x)
            C.obj_fun(O)
        else:
            A, C = C, D
            D.x = A.x + alfa * (B.x - A.x)
            D.obj_fun(O)
        if Solution.f_calls > Nmax or B.x - A.x < epsilon:
            A.x = (A.x + B.x) / 2.0
            A.obj_fun(O)
            return A
        


def Newton(x0, h0, epsilon, Nmax, O):
    X = Solution(x0)
    while True:
        d = -np.linalg.inv(X.h).dot(X.g)
        P = np.column_stack((X.x, d))
        if h0 < 0:
            b = compute_b(X.x, d, O)
            h = golden(0, b, epsilon, Nmax, P)
            X1 = Solution(X.x + h.x * d)
        else:
            X1 = Solution(X.x + h0 * d)
        if np.linalg.norm(X1.x - X.x) < epsilon or Solution.g_calls > Nmax or Solution.f_calls > Nmax or np.linalg.det(X.h) == 0:
            X1.obj_fun()
            print("Newton")
            print("f_calls")
            print(X.f_calls)
            print("g_calls")
            print(X.g_calls)
            print("h_calls")
            print(X.h_calls)
            return X1
        X = X1
def CG(x0, h0, epsilon, Nmax, O):
    X = Solution(x0)
    d = -X.g
    while True:
        P = np.column_stack((X.x, d))
        if h0 < 0:
            b = compute_b(X.x, d, O)
            h = golden(0, b, epsilon, Nmax, P)
            X1 = Solution(X.x + h.x * d)
        else:
            X1 = Solution(X.x + h0 * d)
        if np.linalg.norm(X1.x - X.x) < epsilon or Solution.g_calls > Nmax or Solution.f_calls > Nmax:
            X1.obj_fun()
            # Print calls number
            print("CG")
            print("f_calls")
            print(X.f_calls)
            print("g_calls")
            print(X.g_calls)
            print("h_calls")
            print(X.h_calls)
            return X1
        X1.grad()
        beta = np.linalg.norm(X1.g)**2 / np.linalg.norm(X.g)**2
        d = -X1.g + beta * d
        X = X1
def SD(x0, h0, epsilon, Nmax, O):
    X = Solution(x0)
    while True:
        d = -X.g
        P = np.column_stack((X.x, d))
        if h0 < 0:
            b = compute_b(X.x, d, O)
            h = golden(0, b, epsilon, Nmax, P)
            X1 = Solution(X.x + h.x * d)
        else:
            X1 = Solution(X.x + h0 * d)
        if np.linalg.norm(X1.x - X.x) < epsilon or Solution.g_calls > Nmax or Solution.f_calls > Nmax:
            X1.obj_fun()
            return X1
        X = X1

# Define your initial guess
x0 = np.array([0, 0])

# Define your step size
h0 = 0.1

# Define your tolerance
epsilon = 1e-6

# Define your maximum number of iterations
Nmax = 1000

# Define your matrix O
O = np.array([[0, 1], [1, 0]])

# Call the Newton function
result = Newton(x0, h0, epsilon, Nmax, O)

# Print the result
print(result.x)

result=CG(x0, h0, epsilon, Nmax, O)
print(result.x)

result=SD(x0, h0, epsilon, Nmax, O)
print(result.x)

result=golden(x0, h0, epsilon, Nmax, O)
print(result.x)