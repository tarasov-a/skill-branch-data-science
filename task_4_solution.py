import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

def calculate_data_stats(df): # Задание 1.
    df_shape = df.shape    
    int_count = df.select_dtypes(['float64', 'int64']).dtypes.count()
    obj_count = df.select_dtypes(['object']).dtypes.count()    
    fraud_count = len(df[df['isFraud'] != 0])
    total = len(df)
    fraud_result = round((fraud_count / total) * 100, 2)
    return df_shape, int_count, obj_count, fraud_result

def prepare_data(df): # Задание 2. 
    y = df['isFraud']
    df = df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    return df, y

def fit_first_model(df, y, x_test, y_test): # Задание 3. ТЕСТ
    df = df.fillna(0)
    x_test = x_test.fillna(0)
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=10)
    y_train, y_valid = train_test_split(y, train_size=0.7, shuffle=True, random_state=10)
    model = LogisticRegression(random_state=1)
    model.fit(x_train, y_train)   
    y_pred_proba_valid = model.predict_proba(x_valid)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test)[:, 1]
    valid_score = round(roc_auc_score(y_valid, y_pred_proba_valid), 4)
    test_score = round(roc_auc_score(y_test, y_pred_proba_test), 4)
    return [valid_score, test_score] # [0.5125, 0.5867]

def fit_second_model(df, y, x_test, y_test): # Задание 4.
    df = df.fillna(np.mean(df))
    x_test = x_test.fillna(np.mean(x_test))
    x_train, x_valid = train_test_split(df, train_size=0.75, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=0.75, shuffle=True, random_state=1)
    model = LogisticRegression(random_state=1)
    model.fit(x_train, y_train)    
    y_pred_proba_valid = model.predict_proba(x_valid)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test)[:, 1]
    valid_score = roc_auc_score(y_valid, y_pred_proba_valid)
    test_score = roc_auc_score(y_test, y_pred_proba_test)
    return [round((valid_score), 4), round((test_score), 4)]  

def fit_third_model(df, y, x_test, y_test): # Задание 5.
    df = df.fillna(df.median(axis=0))
    x_test = x_test.fillna(df.median(axis=0))
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=0.7, shuffle=True, random_state=1)
    model = LogisticRegression(random_state=1)
    model.fit(x_train, y_train)    
    y_pred_proba_valid = model.predict_proba(x_valid)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test)[:, 1]
    valid_score = roc_auc_score(y_valid, y_pred_proba_valid)
    test_score = roc_auc_score(y_test, y_pred_proba_test) 
    return [round((valid_score), 4), round((test_score), 4)]    

def fit_fourth_model(df, y, x_test, y_test): # Задание 6-1.    
    df = df.fillna(0)
    x_test = x_test.fillna(0)
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=0.7, shuffle=True, random_state=1)
    scaler = StandardScaler()    
    model = LogisticRegression(random_state=1)
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train, y_train)    
    y_pred_proba_valid = model.predict_proba(x_valid_scaled)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test_scaled)[:, 1]
    valid_score = roc_auc_score(y_valid, y_pred_proba_valid)
    test_score = roc_auc_score(y_test, y_pred_proba_test)
    return [round((valid_score), 4), round((test_score), 4)]

def fit_fifth_model(df, y, x_test, y_test): # Задание 6-2.    
    df = df.fillna(np.mean(df))
    x_test = x_test.fillna(0)
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=0.7, shuffle=True, random_state=1)
    scaler = StandardScaler()    
    model = LogisticRegression(random_state=1)
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train, y_train)    
    y_pred_proba_valid = model.predict_proba(x_valid_scaled)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test_scaled)[:, 1]
    valid_score = roc_auc_score(y_valid, y_pred_proba_valid), 
    test_score = roc_auc_score(y_test, y_pred_proba_test) 
    return [round((valid_score), 4), round((test_score), 4)]

def fit_sixth_model(df, y, x_test, y_test): # Задание 7-1.    
    df = df.fillna(0)
    x_test = x_test.fillna(0)
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=0.7, shuffle=True, random_state=1)
    scaler = MinMaxScaler()    
    model = LogisticRegression(random_state=1)
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train, y_train)   
    y_pred_proba_valid = model.predict_proba(x_valid_scaled)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test_scaled)[:, 1]
    valid_score = roc_auc_score(y_valid, y_pred_proba_valid)
    test_score = roc_auc_score(y_test, y_pred_proba_test)
    return [round((valid_score), 4), round((test_score), 4)]

def fit_seventh_model(df, y, x_test, y_test): # Задание 7-2.    
    df = df.fillna(np.mean(df))
    x_test = x_test.fillna(0)
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=0.7, shuffle=True, random_state=1)
    scaler = MinMaxScaler()  
    model = LogisticRegression(random_state=1)
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train, y_train)    
    y_pred_proba_valid = model.predict_proba(x_valid_scaled)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test_scaled)[:, 1]
    valid_score = roc_auc_score(y_valid, y_pred_proba_valid)
    test_score = roc_auc_score(y_test, y_pred_proba_test)
    return [round((valid_score), 4), round((test_score), 4)]
