import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

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

def fit_first_model(df, y, x_test, y_test): # Задание 3. 
    df = df.fillna(0)
    x_test = x_test.fillna(0)
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=0.7, shuffle=True, random_state=1)
    model = LogisticRegression(random_state=1)
    model.fit(x_train, y_train)   
    y_pred_proba_valid = model.predict_proba(x_valid)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test)[:, 1]
    valid_score = roc_auc_score(y_valid, y_pred_proba_valid)
    test_score = roc_auc_score(y_test, y_pred_proba_test)
    return [round((valid_score), 4), round((test_score), 4)]

def fit_second_model(df, y, x_test, y_test): # Задание 4.
    df = df.fillna(np.mean(df))
    x_test = x_test.fillna(np.mean(x_test))
    x_train, x_valid = train_test_split(df, train_size=0.7, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=0.7, shuffle=True, random_state=1)
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

def check_pipeline(x, y, x_test, y_test, pipeline, split=0.3):
    x_train, x_valid = train_test_split(x, train_size=split, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=split, shuffle=True, random_state=1)
    pipeline.fit(x_train, y_train)
    score1 = roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1])
    score2 = roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1])
    return [round(score1, 4), round(score2, 4)]

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

def find_best_split(x, y, x_test, y_test): # Задание 8.   
    test_sizes = []
    scores1 = []
    scores2 = []
    x_count = x.shape[0]
    for test_size in np.arange(0.1, 1.0, 0.1):
        columns = x.columns
        x2 = x.copy()
        for column in columns:
            median = x2[column].median()
            x2 = x2.fillna(value={column: median})
        x2_test = x_test.copy()
        columns = x2_test.columns
        for column in columns:
            median = x2_test[column].median()
            x2_test = x2_test.fillna(value={column: median})
        pipeline = Pipeline(memory=None,
                            steps=[
                                ('scaling', MinMaxScaler()),
                                ('model', LogisticRegression(random_state=1))
                            ],
                            verbose=False
                            )
        score1,score2 = check_pipeline(x2, y, x2_test, y_test, pipeline, test_size)
        test_sizes.append(int(test_size * x_count))
        scores1.append(score1)
        scores2.append(score2)
    return pd.DataFrame({'Test_size': test_sizes, 'Score1': scores1, 'Score2': scores2})

def choose_best_split(scores): # Задание 9.
    best_test_size = -1
    best_diff = 0
    for index, row in scores.iterrows():
        score1 = row['Score1']
        score2 = row['Score2']
        diff = (score1 - score2)**2

        if best_test_size == -1:
            best_test_size = row['Test_size']
            best_diff = diff
        else:
            if diff < best_diff:
                best_test_size = row['Test_size']
                best_diff = diff
    return int(best_test_size)


