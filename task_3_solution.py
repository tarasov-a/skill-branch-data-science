import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def split_data_into_two_samples(df): # Задание 1. 
    x_train, x_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)
    return x_train, x_test

def prepare_data(df): # Задание 2.
    price_doc = df['price_doc']
    objects = df.select_dtypes(['object'])
    data_x = df.drop(objects, axis=1).drop(['id', 'price_doc'], axis=1).dropna(axis=1)
    return data_x, price_doc

def scale_data(df, transformer): # Задание 3.
    numeric_data_features = df.select_dtypes([np.number]).columns    
    x_scaled = transformer.fit_transform(df[numeric_data_features])
    return pd.DataFrame(x_scaled)    
    
def prepare_data_for_model(df, transformer): # Задание 4.
    price_doc = df['price_doc']
    objects = df.select_dtypes(['object'])
    data_x = df.drop(objects, axis=1).drop(['id', 'price_doc'], axis=1).dropna(axis=1)
    x_scaled = transformer.fit_transform(data_x)     
    return pd.DataFrame(x_scaled), price_doc

def fit_first_linear_model(x_train, y_train): # Задание 5.
    model = LinearRegression()
    model.fit(x_train, y_train)    
    y_train_pred = model.predict(x_train)    
    return y_train_pred 
 


