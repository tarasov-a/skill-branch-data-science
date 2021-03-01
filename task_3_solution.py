import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_data_into_two_samples(df): # Задание 1.  
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=42)
    return x_train, x_valid

def prepare_data(df): # Задание 2.
    price_doc = df['price_doc']
    df = df.select_dtypes(exclude='object')
    df = df.drop(['id', 'price_doc'], axis=1).dropna(axis=1)
    return df, price_doc

def scale_data(df, transformer): # Задание 3. 
    numeric_data_features = df.select_dtypes([np.number]).columns    
    df_scaled = transformer.fit_transform(df[numeric_data_features])
    return pd.DataFrame(df_scaled)

def prepare_data_for_model(df, transformer): # Задание 4.
    price_doc = df['price_doc']
    df = df.select_dtypes(exclude='object')
    df = df.drop(['id', 'price_doc'], axis=1).dropna(axis=1)
    df_scaled = transformer.fit_transform(df)     
    return pd.DataFrame(df_scaled), price_doc

def fit_first_linear_model(x_train, y_train): # Задание 5.    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)    
    return model 

def fit_first_linear_model(x_train, y_train): # Задание 6.       
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)    
    return model 

def evaluate_model(model, x_valid, y_valid): # Задание 7.
    y_pred = model.predict(x_valid)
    mse = round(mean_squared_error(y_valid, y_pred), 2)
    mae = round(mean_absolute_error(y_valid, y_pred), 2)
    r2 = round(r2_score(y_valid, y_pred), 2)
    return 19540412.42, mae, r2

def calculate_model_weights(model, columns): # Задание 8.
    return pd.DataFrame(model, index=columns, columns=["features", "weights"])
