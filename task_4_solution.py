import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
