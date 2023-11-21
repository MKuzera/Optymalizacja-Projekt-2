import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import pdb


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
            # pdb.set_trace()
            x = xb + s * D[i]

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
                        return xb, self.funkcja(xb), self.counter-1
            else:
                s = s * self.alfa

            if np.linalg.norm(s) < self.epsilon or self.counter > self.Nmax:
                return xb, self.funkcja(xb), self.counter-1

# Определение целевой функции
def objective_function(k_values):
    k1, k2 = k_values
    t_span = np.arange(0, 100, 0.1)
    initial_conditions = [0, 0]
    solution = odeint(model, initial_conditions, t_span, args=(k1, k2))
    Q = np.trapz((10 * (np.pi - solution[:, 0])**2 + solution[:, 1]**2 + (k1 * (np.pi - solution[:, 0]) + k2 * solution[:, 1])**2), t_span)
    return Q

# Начальные параметры для оптимизации методом Хука-Дживса
starting_point_hj = np.array([0, 0])
step_length_hj = 0.1
reduction_factor_hj = 0.5
epsilon_hj = 1e-8  # Уменьшено значение epsilon для более точных результатов
max_function_calls_hj = 1000

# Создание объекта оптимизатора с передачей целевой функции
hj_optimizer = HJOptimizer(starting_point_hj, step_length_hj, reduction_factor_hj, epsilon_hj, max_function_calls_hj, objective_function)

# Вызов метода оптимизации
result_hj = hj_optimizer.optimize()

# Вывод результатов оптимизации методом Хука-Дживса
print("Optimal values (Hooke’a-Jeeves):", result_hj)

# Симуляция с оптимальными параметрами
t_span = np.arange(0, 100, 0.1)
initial_conditions_hj = [0, 0]
solution_hj = odeint(model, initial_conditions_hj, t_span, args=(result_hj[0]))

# Построение графиков
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_span, solution_hj[:, 0], label='Hooke’a-Jeeves')
plt.title('Положение манипулятора')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_span, solution_hj[:, 1], label='Hooke’a-Jeeves')
plt.title('Угловая скорость манипулятора')
plt.legend()

plt.tight_layout()
plt.show()
