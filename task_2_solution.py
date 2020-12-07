def calculate_data_shape(x): # Задание 1.
    return x.shape

def take_columns(x): # Задание 2.
    return x.columns

def calculate_target_ratio(x, target_name): # Задание 3.
    return round(np.mean(x[target_name]), 2)

def calculate_data_dtypes(x): # Задание 4.
    return x.dtypes.value_counts()

def calculate_cheap_apartment(x): # Задание 5.
    price = 1000000
    return len(x[x['cost'] < price])


