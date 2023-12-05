import numpy as np
import random

def goal_function(x1, x2):
    first = math.sin( math.pi * ( (x1/math.pi)**2 + (x2/math.pi)**2 )**0.5 )
    second = math.pi * ( (x1/math.pi)**2 + (x2/math.pi)**2 )**0.5
    return first / second
class Vector(object):
    def __init__(self, x, y):
        """ Create a vector, example: v = Vector(1,2) """
        self.x = x
        self.y = y

    def __repr__(self):
        return "({0}, {1})".format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Vector(x, y)

    def __rmul__(self, other):
        x = self.x * other
        y = self.y * other
        return Vector(x, y)

    def __truediv__(self, other):
        x = self.x / other
        y = self.y / other
        return Vector(x, y)

    def c(self):
        return (self.x, self.y)

A = [4, 4.4934, 5]

# objective function
def g1(x1):
    return -x1 + 1

def g2(x2):
    return -x2 + 1

def g3(x1, x2, a):
    return np.sqrt(x1**2 + x2**2) - a

def internal_penalty(x1, x2, a):
    p_1 = g1(x1)
    p_2 = g2(x2)
    p_3 = g3(x1, x2, a)
    p_sum = sum([1/p for p in [p_1, p_2, p_3] if p > 0])
    # print( p_sum )
    return 0 if p_sum == 0 else -1*p_sum

def external_penalty(point, a, lambda_val):
    x1 = point.x
    x2 = point.y
    p_1 = g1(x1)
    p_2 = g2(x2)
    p_3 = g3(x1, x2, a)
    return  lambda_val * sum( [ max(0, p)**2 for p in [p_1, p_2, p_3] if p > 0 ] )

def f(point, lambda_val):
    x1, x2 = point
    first = np.sin(np.pi * np.sqrt((x1/np.pi)**2 + (x2/np.pi)**2))
    second = np.pi * np.sqrt((x1/np.pi)**2 + (x2/np.pi)**2)
    if second == 0:
        result = 0
    else:
        result = first/second
    return result + lambda_val*internal_penalty(x1, x2, A[0])

HISTORY = { 'b': [],
            'g': [],
            'w': [] }
def nelder_mead(alpha=0.5, beta=0.5, gamma=0.2, maxiter=20, lambda_val = 1.0):
    
    # initialization
    v1 = Vector(0, 0)
    v2 = Vector(1.0, 0)
    v3 = Vector(0, 1)

    for _ in range(maxiter):
        adict = {v1:f(v1.c(), lambda_val), v2:f(v2.c(), lambda_val), v3:f(v3.c(), lambda_val)}
        points = sorted(adict.items(), key=lambda x: x[1])
        
        b = points[0][0]
        g = points[1][0]
        w = points[2][0]
        HISTORY['b'].append([b.x, b.y])
        HISTORY['g'].append([g.x, g.y])
        HISTORY['w'].append([w.x, w.y])

        mid = (g + b)/2

        # reflection
        xr = mid + alpha * (mid - w)
        if f(xr.c(), lambda_val) < f(g.c(), lambda_val):
            w = xr
        else:
            if f(xr.c(), lambda_val) < f(w.c(), lambda_val):
                w = xr
            c = (w + mid)/2
            if f(c.c(), lambda_val) < f(w.c(), lambda_val):
                w = c
        if f(xr.c(), lambda_val) < f(b.c(), lambda_val):

            # expansion
            xe = mid + gamma * (xr - mid)
            if f(xe.c(), lambda_val) < f(xr.c(), lambda_val):
                w = xe
            else:
                w = xr
        if f(xr.c(), lambda_val) > f(g.c(), lambda_val):
            
            # contraction
            xc = mid + beta * (w - mid)
            if f(xc.c(), lambda_val) < f(w.c(), lambda_val):
                w = xc
        
        # update points
        v1 = w + Vector(external_penalty(w, A[0], lambda_val), external_penalty(w, A[0], lambda_val))
        v2 = g + Vector(external_penalty(g, A[0], lambda_val), external_penalty(g, A[0], lambda_val))
        v3 = b + Vector(external_penalty(b, A[0], lambda_val), external_penalty(w, A[0], lambda_val))
    return b

print("Result of Nelder-Mead algorithm: ")
xk = nelder_mead()
print(xk)
# print("Best poits is: %s"%(xk))


x_range = (-1, 1)
y_range = (-1, 1)

# Define the list of alpha values to test
alpha_values = [4, 4.9, 5]

# Open the csv file in write mode
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(["x1*", "x2*", "r*", "y*", "Penalty Type", "Penalty Value"])

    for alpha in alpha_values:
        for _ in range(100):
            # Generate a random starting point within the feasible region
            x_start = random.uniform(*x_range)
            y_start = random.uniform(*y_range)
            start_point = Vector(x_start, y_start)

            # Perform the optimization
            result = nelder_mead(alpha=alpha, maxiter=1000, lambda_val = 1.0, v1=start_point)

            # Check if the optimization result is less than 1e-3
            if f(result.c(), 1.0) < 1e-3:
                # After calculating x1*, x2*, r*, y*, write them to the csv file
                x1_star = result.x
                x2_star = result.y
                r_star = math.sqrt(x1_star**2 + x2_star**2)  # distance from origin
                y_star = goal_function(x1_star, x2_star)

                external_penalty_val = external_penalty(result, A[0], lambda_val)
                writer.writerow([x1_star, x2_star, r_star, y_star, "External", external_penalty_val])

                internal_penalty_val = internal_penalty(x1_star, x2_star, A[0])
                writer.writerow([x1_star, x2_star, r_star, y_star, "Internal", internal_penalty_val])

                break