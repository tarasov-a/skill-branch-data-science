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
    
def evaluate_model(model, x_valid, y_valid): # Задание 7.
    y_pred = model.predict(x_valid)
    mse = round(mean_squared_error(y_valid, y_pred), 2)
    mae = round(mean_absolute_error(y_valid, y_pred), 2)
    r2 = round(r2_score(y_valid, y_pred), 2)
    return mse, mae, r2

