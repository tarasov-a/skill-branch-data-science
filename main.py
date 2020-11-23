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
    delta = 0.00001
    lim_x = (function(x + delta, y) - function(x, y))/delta
    lim_y = (function(y + delta, x) - function(y, x))/delta
    lims = [round(lim_x, 2), round(lim_y, 2)]
    return lims

value_2 = gradient(10, 1, function_2)
print(value_2)
      



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
