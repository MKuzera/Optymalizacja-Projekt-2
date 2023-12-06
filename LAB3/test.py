import numpy as np
from scipy.optimize import minimize
import pandas as pd
import math

# Definiowanie funkcji celu
def objective_function(x1, x2):
    return (math.sin(math.pi * math.sqrt((x1/math.pi)**2 + (x2/math.pi)**2)) / (math.pi * math.sqrt((x1/math.pi)**2 + (x2/math.pi)**2)))
# Definiowanie ograniczeń
def g1(x1):
    return -x1 + 1
def g2(x2):
    return -x2 + 1
def g3(x1, x2, a):
    return math.sqrt(x1**2 + x2**2) - a
# Definiowanie funkcji kary wewnętrznej


# wzor wewnetrzna funkcja kary − ∑
# 1
# 𝑔𝑖
# (𝑥1,𝑥2
# )
# 𝑛
# 𝑖=1
def inner_penalty_function(x1, x2, alpha):
    return -1 * sum([1/g for g in [g1(x1), g2(x2), g3(x1, x2, alpha)] if g > 0])
#wzor zewnetrzna funkcja kary
# ∑i=1 (𝑚𝑎𝑥(0, 𝑔𝑖
# (𝑥1, 𝑥2
# )))
# 2 
def outer_penalty_function(x1, x2, alpha):
    return sum([max(0, g)**2 for g in [g1(x1), g2(x2), g3(x1, x2, alpha)] if g > 0])

# Definiowanie funkcji celu z uwzględnieniem kar
def penalized_objective_function(x):
    return objective_function(x) + inner_penalty_function(x) + outer_penalty_function(x)

# Początkowy punkt

alphas = [4, 4.4934, 5]
for alpha in alphas:
    results_inner = []
    results_outer = []
    for _ in range(100):
        x0 = np.random.uniform(-1, 1, 2)

        def penalized_objective_function_inner(x):
            return objective_function(x[0], x[1]) + inner_penalty_function(x[0], x[1], alpha)

        def penalized_objective_function_outer(x):
            return objective_function(x[0], x[1]) + outer_penalty_function(x[0], x[1], alpha)

        res_inner = minimize(penalized_objective_function_inner, x0, method='nelder-mead', options={'xatol': 1e-3, 'disp': True})
        res_outer = minimize(penalized_objective_function_outer, x0, method='nelder-mead', options={'xatol': 1e-3, 'disp': True})

        results_inner.append([x0[0], x0[1], res_inner.x[0], res_inner.x[1], res_inner.fun, np.linalg.norm(res_inner.x), res_inner.nfev])
        results_outer.append([x0[0], x0[1], res_outer.x[0], res_outer.x[1], res_outer.fun, np.linalg.norm(res_outer.x), res_outer.nfev])

    df_inner = pd.DataFrame(results_inner, columns=['x10', 'x20', 'x1*', 'x2*', 'y*', 'r*', 'nfev'])
    df_outer = pd.DataFrame(results_outer, columns=['x10', 'x20', 'x1*', 'x2*', 'y*', 'r*', 'nfev'])

    df_inner.to_csv(f'inner_alpha_{alpha}.csv', index=False)
    df_outer.to_csv(f'outer_alpha_{alpha}.csv', index=False)
