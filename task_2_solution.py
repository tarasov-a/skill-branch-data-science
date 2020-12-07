import pandas as pd
import numpy as np


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

def calculate_squad_in_cheap_apartment(x): # Задание 6.
    price = 1000000
    cheap_ap = x[x['cost'] < price]
    return round(np.mean(cheap_ap['full_sq']))

def calculate_mean_price_in_new_housing(x): # Задание 7.
    rooms = 3
    year = 2010
    new_ap = x[(x['rooms'] == rooms) & (apartments['year'] >= year)]
    return round(np.mean(new_ap['cost']))
