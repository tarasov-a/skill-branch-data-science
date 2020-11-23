import numpy as np
import math

def function_1(x): 
    return np.cos(x) + 0.05*(x**3) + np.log2(x**2)

def derivation(x, function):
    delta_x = 1e-10
    lim_x = (function(x + delta_x) - function(x))/delta_x
    return lim_x

value_1 = derivation(10, function_1)
print(round(value_1, 2))

