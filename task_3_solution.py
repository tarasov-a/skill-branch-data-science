import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from copy import deepcopy

from sklearn.datasets import make_regression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def split_data_into_two_samples(df):
    return train_test_split(df,test_size = 0.3, train_size = 0.7, shuffle=True, random_state = 42)
    
def prepare_data(df_main):
    df = df_main.copy()
    for x in df.columns:
        if df[x].dtype == 'object':
            df.drop(x, axis = 1, inplace = True)
    df.drop('id', axis = 1, inplace = True)
    df.dropna(axis = 1, inplace = True)
    g_var = df['price_doc']
    df.drop('price_doc', axis = 1, inplace = True)
    return [df, g_var]
    
def scale_data(df_main, scaler = StandardScaler()):
    df = df_main
    scaler.fit(df)
    return scaler.transform(df)
    
def prepare_data_for_model(df_main, scaler = StandardScaler()):
    [df, g_var] = prepare_data(df_main)
    return [scale_data(df, scaler), g_var]
    
def fit_first_linear_model(x_train, y_train):
    [x_train, y_train, features] = prepare_data_for_model(d_train)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def fit_first_linear_model_cop(x_train, y_train):
    [x_train, y_train, features] = prepare_data_for_model(d_train, MinMaxScaler())
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model
    
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2 = round(r2_score(y_test, y_pred), 2)
    return [mse, mae, r2]
    
def calculate_model_weights(model, features):
    sorted_weights = sorted(zip(model.coef_, features), reverse=True)
    weights = pd.Series([x[0] for x in sorted_weights])
    features = pd.Series([x[1] for x in sorted_weights])
    df = pd.DataFrame({'features': features, 'weights': weights})
    return df
