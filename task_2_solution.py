import pandas as pd
import numpy as np


def calculate_data_shape(x): # Задание 1.
    return x.shape

def take_columns(x): # Задание 2.
    return x.columns

def calculate_target_ratio(x, target_name): # Задание 3.
    return round(np.mean(x[target_name]), 2)

def calculate_data_dtypes(x): # Задание 4. 
    types = x.dtypes.value_counts()    
    return types[0] + types[1], types[2]  

def calculate_cheap_apartment(x): # Задание 5.
    price = 1000000
    return len(x[x['price_doc'] <= price])

def calculate_squad_in_cheap_apartment(x): # Задание 6.
    price = 1000000
    cheap_ap = x[x['price_doc'] <= price]
    return round(np.mean(cheap_ap['full_sq']))

def calculate_mean_price_in_new_housing(x): # Задание 7.
    rooms = 3
    year = 2010
    new_ap = x[(x['num_room'] == rooms) & (x['build_year'] >= year)]
    return round(np.mean(new_ap['price_doc']))

def calculate_mean_squared_by_num_rooms(x): # Задание 8.
    mean_sq = x.groupby(['num_room'])['full_sq'].mean()
    return round(mean_sq, 2)  

def calculate_squared_stats_by_material(x): # Задание 9.
    mean_sq = x.groupby(['material'])['full_sq'].mean()
    return round(mean_sq, 2

def calculate_crosstab(x): # Задание 10.
    mean_price = x.pivot_table(index='sub_area', values='price_doc', columns='product_type', aggfunc='mean', fill_value=0)
    return round(mean_price, 2)

