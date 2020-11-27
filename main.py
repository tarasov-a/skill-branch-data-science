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



def function_2(x1, x2): # функция 2
    return x1**2*np.cos(x2) + 0.05*(x2)**3 + 3*(x1)**3*np.log2(x2**2)

list_v = [x1, x2]

x1 = 10
x2 = 1

def gradient(list_of_values, function): # задание 2
    list_lims = []    
    delta = 0.00001
    lim_x = (function(x1 + delta, x2) - function(x1, x2)) / delta
    list_lims.append(round(lim_x, 2))
    lim_y = (function(x1, x2 + delta) - function(x1, x2)) / delta
    list_lims.append(round(lim_y, 2))
    return list_lims


value_3 = gradient(list_v, function_2)
print(value_3)


def gradient_optimization_one_dim(function): # задание 3
    x_0 = 10 # начальная позиция
    e = 0.001 # шаг
    N = 50 # количество итераций
    
    for i in range(N):
        x_0 = x_0 - e*derivation(x_0, function)
    return round(x_0, 2)

value_3 = gradient_optimization_one_dim(function_1)
print(value_3)
