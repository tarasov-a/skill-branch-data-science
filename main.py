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


def gradient_optimization_one_dim(function): # функция градиентного спуска      
    x_0 = 10 # начальная позиция
    e = 0.001 # шаг
    N = 50 # количество итераций
    
    for i in range(N):
        x_0 = x_0 - e*derivation(x_0, function)
    return round(x_0, 2)

value_3 = gradient_optimization_one_dim(function_1)
print(value_3)
