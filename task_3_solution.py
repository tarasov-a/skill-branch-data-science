import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_data_into_two_samples(df): # Задание 1.  
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=42)
    y_train, y_valid = train_test_split(df['price_doc'], train_size=0.7, shuffle=True, random_state=42)
    return x_train, y_train, x_valid, y_valid
