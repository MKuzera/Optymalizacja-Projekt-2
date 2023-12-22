import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize_scalar


class Lab4:
    TEST_RUN_COUNT = 0
    GRADIENT_RUN_COUNT = 0
    HESSIAN_RUN_COUNT = 0
    ITERATION_COUNT = 0
    ITERATION_COUNT = 0
    RESULT = {'x1': [], 'x2': [], 'f_calls': [], 'g_calls': [], 'h_calls': [], 'function': [], 'step': []}
    ITERATION_DATA= {'x1': [], 'x2': [], 'iter_number': [], 'step':[]}
    
    def write_dict(self, x1, x2, iter, step):
        Lab4.ITERATION_DATA['x1'].append(x1)
        Lab4.ITERATION_DATA['x2'].append(x2)
        Lab4.ITERATION_DATA['iter_number'].append(iter)
        Lab4.ITERATION_DATA['step'].append(step)
    
    def testowa_funkcja(self, x):
        Lab4.TEST_RUN_COUNT+=1
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def gradient_testowej_funkcji(self, x):
        Lab4.GRADIENT_RUN_COUNT+=1
        df_dx1 = 2 * (x[0] + 2*x[1] - 7) + 4 * (2*x[0] + x[1] - 5)
        df_dx2 = 4 * (x[0] + 2*x[1] - 7) + 2 * (2*x[0] + x[1] - 5)
        return np.array([df_dx1, df_dx2])

    def conjugate_gradients(self, starting_point, tolerance, max_iterations, fixed_step=True, step_size=0.05):
        x = starting_point
        gradient = self.gradient_testowej_funkcji(x)
        direction = -gradient

        for i in range(max_iterations):
            Lab4.ITERATION_COUNT+=1
            gradient = self.gradient_testowej_funkcji(x)
            beta = np.dot(gradient, gradient) / np.dot(direction, direction)
            self.write_dict(x[0], x[1], i, str(step_size))
            
            if np.abs(beta) < 1e-8:
                break
            if fixed_step:
                x = x + step_size * direction
            else:
                res = minimize_scalar(lambda alpha: self.testowa_funkcja(x + alpha * direction))
                x = x + res.x * direction

            new_direction = -gradient + beta * direction
            direction = new_direction

            if np.linalg.norm(gradient) < tolerance:
                break

        return x
    
    def steepest_descent(self, starting_point, tolerance, max_iterations, fixed_step=True, step_size=0.05):
        # najzybszy spadek
        x = starting_point
        for i in range(max_iterations):
            Lab4.ITERATION_COUNT+=1
            self.write_dict(x[0], x[1], i, 'M.zk.' if fixed_step else  str(step_size))
            gradient = self.gradient_testowej_funkcji(x)
            
            if fixed_step:
                x = x - step_size * gradient
            else:
                res = minimize_scalar(lambda alpha: self.testowa_funkcja(x - alpha * gradient))
                x = x - res.x * gradient

            if np.linalg.norm(gradient) < tolerance:
                break
        return x
    
    def hessian_testowej_funkcji(self, x):
        Lab4.HESSIAN_RUN_COUNT+=1
        d2f_dx1dx1 = 2 + 8
        d2f_dx1dx2 = 4
        d2f_dx2dx1 = 4
        d2f_dx2dx2 = 4 + 2
        return np.array([[d2f_dx1dx1, d2f_dx1dx2], [d2f_dx2dx1, d2f_dx2dx2]])

    def newton_method(self, starting_point, tolerance, max_iterations, fixed_step=True, step_size=0.05):
        x = starting_point

        for i in range(max_iterations):
            Lab4.ITERATION_COUNT+=1
            self.write_dict(x[0], x[1], i, 'M.zk.' if fixed_step else  str(step_size))

            gradient =  self.gradient_testowej_funkcji(x)
            hessian_inv = inv(self.hessian_testowej_funkcji(x))

            if fixed_step:
                x = x - step_size * np.dot(hessian_inv, gradient)
            else:
                res = minimize_scalar(lambda alpha: self.testowa_funkcja(x - alpha * np.dot(hessian_inv, gradient)))
                x = x - res.x * np.dot(hessian_inv, gradient)

            if np.linalg.norm(gradient) < tolerance:
                break

        return x
    
    def dict_to_csv(self, dict, file_name):
        df = pd.DataFrame(dict)
        df.to_csv(f'wykresy/iteration_{file_name}.csv', index=False)
        
    
    def zeroing_out(self):
        Lab4.TEST_RUN_COUNT = 0
        Lab4.GRADIENT_RUN_COUNT = 0
        Lab4.HESSIAN_RUN_COUNT = 0
        Lab4.ITERATION_COUNT = 0
        
    def clear_iteration_data(self):
        Lab4.ITERATION_DATA= {'x1': [], 'x2': [], 'iter_number': [], 'step':[]}
    
    def generate_random_numbers():
        x1 = np.random.uniform(-10, 10)
        x2 = np.random.uniform(-10, 10)
        return x1, x2
    
    def start_calculation(self):
        pd_diict = {'start_x1': [],'start_x2':[], 'step': [],
                    'x1_steepest_descent':[], 'x2_steepest_descent':[], 'y_steepest_descent': [],'f_calls_steepest_descent':[], 'g_calls_steepest_descent':[], 'h_calls_steepest_descent':[], 
                    'x1_conjugate_gradients':[], 'x2_conjugate_gradients':[], 'y_conjugate_gradients':[], 'f_calls_conjugate_gradients':[], 'g_calls_conjugate_gradients':[], 'h_calls_conjugate_gradients':[], 
                    'x1_newton_method':[], 'x2_newton_method':[], 'y_newton_method':[], 'f_calls_newton_method':[], 'g_calls_newton_method':[], 'h_calls_newton_method':[], }
        
        tolerance = 1e-5
        max_iterations = 1000
        for _ in range(100):
            x1, x2 = Lab4.generate_random_numbers()
            starting_point = np.array([0.0, 0.0])
            for step in [['0,12', 0.12], ['0,05',0.05], ['M.zk.', 0.05]]:
                fixed = True if step[0] == '0,12' or step[0] == '0,05' else False
                pd_diict['start_x1'].append(x1)
                pd_diict['start_x2'].append(x2)
                pd_diict['step'].append(step[0])
                
                result_des = self.steepest_descent(starting_point, tolerance, max_iterations, fixed, step[1])
                Lab4.RESULT['x1'].append(result_des[0])
                Lab4.RESULT['x2'].append(result_des[1])
                Lab4.RESULT['f_calls'].append(Lab4.TEST_RUN_COUNT)
                Lab4.RESULT['g_calls'].append(Lab4.GRADIENT_RUN_COUNT)
                Lab4.RESULT['h_calls'].append(Lab4.HESSIAN_RUN_COUNT)
                Lab4.RESULT['function'].append('steepest_descent')
                Lab4.RESULT['step'].append(step[0])

                self.dict_to_csv(Lab4.ITERATION_DATA, f'steepest_descent_{step[0]}')
                self.clear_iteration_data()
                pd_diict['x1_steepest_descent'].append(result_des[0])
                pd_diict['x2_steepest_descent'].append(result_des[1])
                pd_diict['y_steepest_descent'].append(self.testowa_funkcja(result_des))
                pd_diict['f_calls_steepest_descent'].append(Lab4.TEST_RUN_COUNT)
                pd_diict['g_calls_steepest_descent'].append(Lab4.GRADIENT_RUN_COUNT)
                pd_diict['h_calls_steepest_descent'].append(Lab4.HESSIAN_RUN_COUNT)

                
                self.zeroing_out()
                
                result_grad = self.conjugate_gradients(starting_point, tolerance, max_iterations, fixed, step[1])
                Lab4.RESULT['x1'].append(result_des[0])
                Lab4.RESULT['x2'].append(result_des[1])
                Lab4.RESULT['f_calls'].append(Lab4.TEST_RUN_COUNT)
                Lab4.RESULT['g_calls'].append(Lab4.GRADIENT_RUN_COUNT)
                Lab4.RESULT['h_calls'].append(Lab4.HESSIAN_RUN_COUNT)
                Lab4.RESULT['function'].append('conjugate_gradients')
                Lab4.RESULT['step'].append(step[0])

                
                self.dict_to_csv(Lab4.ITERATION_DATA, f'conjugate_gradients_{step[0]}')
                self.clear_iteration_data()
                pd_diict['x1_conjugate_gradients'].append(result_grad[0])
                pd_diict['x2_conjugate_gradients'].append(result_grad[1])
                pd_diict['y_conjugate_gradients'].append(self.testowa_funkcja(result_grad))
                pd_diict['f_calls_conjugate_gradients'].append(Lab4.TEST_RUN_COUNT)
                pd_diict['g_calls_conjugate_gradients'].append(Lab4.GRADIENT_RUN_COUNT)
                pd_diict['h_calls_conjugate_gradients'].append(Lab4.HESSIAN_RUN_COUNT)
                self.zeroing_out()
                
                result_newton = self.newton_method(starting_point, tolerance, max_iterations, fixed, step[1])
                Lab4.RESULT['x1'].append(result_des[0])
                Lab4.RESULT['x2'].append(result_des[1])
                Lab4.RESULT['f_calls'].append(Lab4.TEST_RUN_COUNT)
                Lab4.RESULT['g_calls'].append(Lab4.GRADIENT_RUN_COUNT)
                Lab4.RESULT['h_calls'].append(Lab4.HESSIAN_RUN_COUNT)
                Lab4.RESULT['function'].append('newton_method')
                Lab4.RESULT['step'].append(step[0])
                
                self.dict_to_csv(Lab4.ITERATION_DATA, f'newton_method_{step[0]}')
                
                self.clear_iteration_data()
                pd_diict['x1_newton_method'].append(result_newton[0])
                pd_diict['x2_newton_method'].append(result_newton[1])
                pd_diict['y_newton_method'].append(self.testowa_funkcja(result_newton))
                pd_diict['f_calls_newton_method'].append(Lab4.TEST_RUN_COUNT)
                pd_diict['g_calls_newton_method'].append(Lab4.GRADIENT_RUN_COUNT)
                pd_diict['h_calls_newton_method'].append(Lab4.HESSIAN_RUN_COUNT)
                self.zeroing_out()
                
        for key in Lab4.RESULT.keys():
            print(key, len(Lab4.RESULT[key]))
        
        pd.DataFrame(Lab4.RESULT).to_csv('results.csv', index=False)
        return pd_diict
        
if __name__ == '__main__':
    res_dict = Lab4().start_calculation()
    df = pd.DataFrame(res_dict)