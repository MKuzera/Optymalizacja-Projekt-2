import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Constants
m = 0.6  # kg
r = 0.12  # m
g = 9.81  # m/s^2
C = 0.47
rho = 1.2  # kg/m^3
S = np.pi * r**2

# Equations of motion
def equations(t, y, vx0, omega):
    x, vx, y, vy = y
    Dx = 0.5 * C * rho * S * vx**2
    Dy = 0.5 * C * rho * S * vy**2
    FMx = rho * vy * omega * np.pi * r**3
    FMy = rho * vx * omega * np.pi * r**3
    return [vx, -Dx/m - FMx/m, vy, -Dy/m - FMy/m - g]

# Objective function
def objective(params):
    vx0, omega = params
    sol = solve_ivp(equations, [0, 7], [0, vx0, 100, 0], args=(vx0, omega), t_eval=np.arange(0, 7, 0.01))
    return -sol.y[0, -1]

# Constraint
def constraint(params):
    vx0, omega = params
    sol = solve_ivp(equations, [0, 7], [0, vx0, 100, 0], args=(vx0, omega), t_eval=np.arange(0, 7, 0.01))
    x_at_y_50 = np.interp(50, sol.y[2, :], sol.y[0, :])
    return (4 - x_at_y_50)**2 + (x_at_y_50 - 6)**2 - 1

# Optimization
result = minimize(objective, [0, 0], method='Nelder-Mead', constraints={'type': 'ineq', 'fun': constraint}, bounds=[(-10, 10), (-23, 23)])

print("Optimal initial horizontal velocity (m/s):", result.x[0])
print("Optimal initial rotation rate (rad/s):", result.x[1])
import pandas as pd

# Create a DataFrame for the results
df = pd.DataFrame([result.x], columns=['vx0', 'omega'])
df['vx0'] = df['vx0'].map('{:.2f}'.format)
df['omega'] = df['omega'].map('{:.2f}'.format)
print(df)

# Run the simulation with the optimal values
sol = solve_ivp(equations, [0, 7], [0, result.x[0], 100, 0], args=(result.x[0], result.x[1]), t_eval=np.arange(0, 7, 0.01))

# Create a DataFrame for the simulation results
df_sim = pd.DataFrame({'t': sol.t, 'x': sol.y[0, :], 'y': sol.y[2, :]})
# Save the simulation results to a CSV file
df_sim.to_csv('simulation_results.csv', index=False)
print(df_sim)