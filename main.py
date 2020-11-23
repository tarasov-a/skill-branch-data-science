def derivation (x: float, func: callable) -> float:
    delta = 1e-5
    x_left, x_right = func(x), func(x + delta)
    value = (x_right - x_left) / delta
    return value

def func_first(x):
    return np.cos(x) + 0.05*x**3 + np.log2(x**2)

value = derivation(10, func_first)
print(round(value, 2))

