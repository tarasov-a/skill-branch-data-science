import numpy as np
import math

def function_1(x):   
    return np.cos(x) + 0.05*(x**3) + np.log2(x**2)

def derivation(x, function):      
    delta_x = 1e-10
    lim_x = (function(x + delta_x) - function(x)) / delta_x
    return round(lim_x, 2)

value_1 = derivation(10, function_1)
print(value_1)


def gradient_optimization_one_dim(function, x = 10, max_iterations = 50, epsilone = 0.001):
    while max_iterations:
        max_iterations -= 1
        x = x - epsilone * derivation(x, function)
    return round(x, 2)

value_3 = gradient_optimization_one_dim(function_1)
print(value_3)
