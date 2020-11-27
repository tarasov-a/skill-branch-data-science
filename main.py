import numpy as np
def function_1(x): 
    return np.cos(x) + 0.05*(x**3) + np.log2(x**2)

def derivation(x, function):
    delta_x = 0.00001  
    lim_x = (function(x + delta_x) - function(x))/delta_x
    return lim_x

value_1 = derivation(10, function_1)
print(round(value_1, 2))

def function_2(x, y): 
    return x**2*np.cos(y) + 0.05*(y)**3 + 3*(x)**3*np.log2(y**2)


def gradient(x, y, function):
    list_lims = []
    delta = 0.00001
    lim_x = (function(x + delta, y) - function(x, y)) / delta
    list_lims.append(round(lim_x, 2))
    lim_y = (function(x, y + delta) - function(x, y)) / delta
    list_lims.append(round(lim_y, 2))
    return list_lims

value_3 = diff(10, 1, function_2)
print(value_3)
      



def gradient_optimization_one_dim(function): # функция градиентного спуска
    delta_x = 0.00001  
    x_0 = 10 # начальная позиция
    e = 0.001 # шаг
    N = 50 # количество итераций
    lim_x = (function_1(x_0 + delta_x) - function_1(x_0))/delta_x
    for i in range(N):
        x_0 = x_0 - e*lim_x
    return x_0
        
    
value_3 = gradient_optimization_one_dim(function_1)
print(round(value_3, 2))
