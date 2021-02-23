import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def split_data_into_two_samples(x): # Задание 1.
    x_train, x_test = train_test_split(x, test_size=0.3, shuffle=True, random_state=42)
    return x_train, x_test

def prepare_data(x):
    price_doc = x['SalePrice']
    objects = x.select_dtypes(['object'])
    data_x = x.drop(objects, axis=1).drop(['Id'], axis=1).dropna(axis=1)
    return data_x, price_doc


