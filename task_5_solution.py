import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def prepare_data(df): # Задание 1.
    target = df['isFraud']
    df_dropped = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    return df_dropped, target

def fit_first_model(x, y, x_test, y_test): # Задание 2. 
    x = x.fillna(0)
    x_test = x_test.fillna(0)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.7, random_state=1)
    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(x_train, y_train)
    valid_pred = tree.predict_proba(x_valid)
    score = roc_auc_score(y_valid, valid_pred[:,1])
    return round(score, 4) 

def fit_second_model(x, y, x_test, y_test): # Задание 3.
    x = x.fillna(-9999)
    x_test = x_test.fillna(-9999)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.7, random_state=1)
    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(x_train, y_train)
    valid_pred = tree.predict_proba(x_valid)
    score = roc_auc_score(y_valid, valid_pred[:,1])
    return round(score, 4)
