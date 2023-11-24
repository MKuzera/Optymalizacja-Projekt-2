import math
import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2 - math.cos(2.5*math.pi*x[0]) - math.cos(2.5*math.pi*x[1]) + 2

# Initial values
x0 = np.random.rand(2)

# Hooke's-Jeevesa method
result_HJ = minimize(objective_function, x0, method='Powell', options={'xtol': 1e-6, 'maxiter': 100})

# Rosenbrocka method
result_RB = minimize(objective_function, x0, method='Nelder-Mead', options={'xtol': 1e-6, 'maxiter': 100})

# Display the results in a table
print("Nr iteracji\tMetoda Hooke’a-Jeevesa\t\tMetoda Rosenbrocka")
print("x1*\t\tx2*\t\tx1*\t\tx2*")

for i in range(result_HJ.nit):
    print(f"{i}\t\t{result_HJ.x[0]:.8f}\t{result_HJ.x[1]:.8f}\t{result_RB.x[0]:.8f}\t{result_RB.x[1]:.8f}")

