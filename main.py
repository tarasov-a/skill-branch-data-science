import numpy as np
import math

def function_1(x):  # функция 1
    return np.cos(x) + 0.05*(x**3) + np.log2(x**2)

def derivation(x, function):  # задание 1    
    delta_x = 1e-10
    lim_x = (function(x + delta_x) - function(x)) / delta_x
    return round(lim_x, 2)

value_1 = derivation(10, function_1)
print(value_1)



def f(x1, x2): # функция 2
    return x1**2*np.cos(x2) + 0.05*(x2)**3 + 3*(x1)**3*np.log2(x2**2)



def gradient(list_X, f):
    
    values = []
    delta = 0.0001
    x1 = list_X[0]
    x2 = list_X[1]   
    F = (f(x1+delta, x2) - f(x1, x2))/delta
    values.append(round(F, 2))
    F = (f(x1, x2+delta) - f(x1, x2))/delta
    values.append(round(F, 2))
    return values


def gradient_optimization_one_dim(function): # задание 3
    x_0 = 10 # начальная позиция
    e = 0.001 # шаг
    N = 50 # количество итераций
    
    for i in range(N):
        x_0 = x_0 - e*derivation(x_0, function)
    return round(x_0, 2)

value_3 = gradient_optimization_one_dim(function_1)
print(value_3)
