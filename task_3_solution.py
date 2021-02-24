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

def scale_data(x, type_of_scaler): # Задание 3
    numeric_data_features = x.select_dtypes([np.number]).columns
    if type_of_scaler == MinMaxScaler:        
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x[numeric_data_features])
        return x_scaled
    if type_of_scaler == StandardScaler:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x[numeric_data_features])
        return x_scaled


