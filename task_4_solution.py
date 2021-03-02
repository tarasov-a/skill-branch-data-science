import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline


# Задание 1
# Написать функцию `calculate_data_stats`, которая принимает на вход датафрейм с данными и возвращает
# размерность набора данных, количество числовых признаков, количество категориальных признаков и
# долю целевого события (`isFraud`). Долю целевого события округлить до 2-го знака.
def calculate_data_stats(x):
    shape = x.shape
    ints = x.select_dtypes(include=['float64', 'int64']).dtypes.count()
    objects = x.select_dtypes(include=['object']).dtypes.count()
    is_fraud_positive = x[x['isFraud'] == True]['isFraud'].count()
    is_fraud_total = x['isFraud'].count()
    fraud_share = (is_fraud_positive * 100 / is_fraud_total).round(2)
    return [shape, ints, objects, fraud_share]


# Задание 2
# Наш набор данных содержит служебную информацию, которая нам не нужна при обучении модели.
# Написать функцию `prepare_data`, которая принимает на вход датафрейм, в теле функции выделяет целевую переменную
# `isFraud` в отдельный вектор и удаляет ненужные для обучения признаки: `isFraud`, `TransactionID`, `TransactionDT`.
# Функция должна возвращать преобразованный набор данных и вектор целевой переменной.
def prepare_data(x):
    target_vector = x['isFraud']
    filtered = x.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    return [filtered, target_vector]


def build_model(x, y, x_test, y_test, split=0.3):
    x_train, x_valid = train_test_split(x, train_size=split, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=split, shuffle=True, random_state=1)

    model = LogisticRegression(random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_valid)[:, 1]
    score1 = roc_auc_score(y_valid, y_pred)

    y_test_pred = model.predict_proba(x_test)[:, 1]
    score2 = roc_auc_score(y_test, y_test_pred)

    return [round(score1, 4), round(score2, 4)]


# Задание 3
# Начнем обучать модель и сравнивать их качество.
# Основная схема будет выглядеть следующим образом - будем разбивать выборку `train` на обучение / контроль / тест,
# обучать и оценивать качество модели на этих выборках, а в конце задания будем делать предсказания
# на отдельной выборке `test`. Будем сранвивать качество модели на нашей локальной валидации и на выборке `test`.
# Для оценки качества модели будем использовать метрику - __ROC_AUC__
#
# Написать функцию `fit_first_model`, которая принимает аргументы `X`, `y`, где `X` - набор данных для обучения,
# `y` - вектор целевой переменной, а также `x_test` и `y_test`. Разбить исходную выборку на `x_train`, `x_valid`
# в соотношении 70/30, зафиксированть параметр `random_state=1` и перемешать данные.
# После разбиения обучить логистическую регрессию с параметром `random_state=1` на `x_train`,
# и оценить качество модели на `x_valid`. При необходимости, заполнить пропуски 0.
# Функция должна возвращать оценку качества модели на выборках `x_valid` и `x_test`.
# Значение метрики качества оценить с точностью до 4-го знака.
#
# __Пример__:
# `def fit_first_model(X, y, x_test, y_test) -> Tuple[float, float]:`
def fit_first_model(x, y, x_test, y_test):
    x2 = x.fillna(0)
    x2_test = x_test.fillna(0)
    return build_model(x2, y, x2_test, y_test)


# Задание 4
# мы обучили нашу первую модель, но почему мы используем заполнение пропусков 0? Давай попробуем заполнить
# пропуски средним значением по каждому отдельному признаку и выполнить задание 3 еще раз. Среднее посчитать
# по той выборке, которая используется для определенного действия (если обучение модели - то по обучающей,
# если тестирование модели - то по тестовой). Функцию для этого задания, назовем `fit_second_model`.
# Устно проанализируйте изменения качества модели, стала модель лучше? Если да, то почему? Этот анализ сделать устно,
# ответы на эти вопросы сдавать автоматизированной системе не требуется.
def fit_second_model(x, y, x_test, y_test):
    columns = x.columns
    x2 = x.copy()
    for column in columns:
        mean = x2[column].mean()
        x2 = x2.fillna(value={column: mean})
    columns = x_test.columns
    x_test2 = x_test.copy()
    for column in columns:
        mean = x_test2[column].mean()
        x_test2 = x_test2.fillna(value={column: mean})

    return build_model(x2, y, x_test2, y_test)


# Задание 5
# В задании на линейную регрессию, мы говорили, что среднее - статистика, которая сильно зависит от выбросов,
# она ориентируется на выбросы. А вот медиана - статистика, которая более устойчива к выбросам.
# Обработаем пропуски медианой, и выполним задание 3. Функцию назовем `fit_third_model`.
def fit_third_model(x, y, x_test, y_test):
    columns = x.columns
    for column in columns:
        median = x[column].median()
        x = x.fillna(value={column: median})
    columns = x_test.columns
    for column in columns:
        median = x_test[column].median()
        x_test = x_test.fillna(value={column: median})
    return build_model(x, y, x_test, y_test)


# Задание 6
# Линейные модели сильно зависят от масштаба признаков. Наши данные содержат признаки в разном масштабе.
# Давай попробуем отмасштабировать данные с помощью `StandardScaler`.
# Для удобства, можно создать единый пайплайн, который и предобработает данные, и обучит модель логистической регрессии.
# Выполнить задание 3 - функция `fit_fourth_model` и задание 4 - функция `fit_fifth_model`.
def check_pipeline(x, y, x_test, y_test, pipeline, split=0.3):
    x_train, x_valid = train_test_split(x, train_size=split, shuffle=True, random_state=1)
    y_train, y_valid = train_test_split(y, train_size=split, shuffle=True, random_state=1)
    pipeline.fit(x_train, y_train)
    score1 = roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1])
    score2 = roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1])
    return [round(score1, 4), round(score2, 4)]


def fit_fourth_model(x, y, x_test, y_test):
    x = x.fillna(0)
    x_test = x_test.fillna(0)

    pipeline = Pipeline(memory=None,
             steps=[
                 ('scaling', StandardScaler()),
                 ('model', LogisticRegression(random_state=1))
             ],
             verbose=False
    )
    return check_pipeline(x, y, x_test, y_test, pipeline)


def fit_fifth_model(x, y, x_test, y_test):
    columns = x.columns
    for column in columns:
        mean = x[column].mean()
        x = x.fillna(value={column: mean})

    columns = x_test.columns
    for column in columns:
        mean = x_test[column].mean()
        x_test = x_test.fillna(value={column: mean})
    pipeline = Pipeline(memory=None,
                        steps=[
                            ('scaling', StandardScaler()),
                            ('model', LogisticRegression(random_state=1))
                        ],
                        verbose=False
                        )
    return check_pipeline(x, y, x_test, y_test, pipeline)


# Задание 7
# Для масштабирования данные есть и другой инструмент - `MinMaxScaler`. Давай попробуем выполнить масштабирование
# с помощью этого объекта и оценим качество модели. Выполнить задание 3 - функция `fit_sixth_model`
# и задание 4 - функция `fit_seventh_model`.
def fit_sixth_model(x, y, x_test, y_test):
    x = x.fillna(0)
    x_test = x_test.fillna(0)
    pipeline = Pipeline(memory=None,
                        steps=[
                            ('scaling', MinMaxScaler()),
                            ('model', LogisticRegression(random_state=1))
                        ],
                        verbose=False
                        )
    return check_pipeline(x, y, x_test, y_test, pipeline)


def fit_seventh_model(x, y, x_test, y_test):
    columns = x.columns
    for column in columns:
        mean = x[column].mean()
        x = x.fillna(value={column: mean})

    columns = x_test.columns
    for column in columns:
        mean = x_test[column].mean()
        x_test = x_test.fillna(value={column: mean})

    pipeline = Pipeline(memory=None,
                        steps=[
                            ('scaling', MinMaxScaler()),
                            ('model', LogisticRegression(random_state=1))
                        ],
                        verbose=False
                        )
    return check_pipeline(x, y, x_test, y_test, pipeline)


# Задание 8
# Отлично, мы провели ряд экспериментов с разными подходами к подготовки данных.
# Теперь поэкспериментируем с разным разбиением исходной выборки на обучающую и валидационную часть,
# и поймем какая комбинация является оптимальной. Написать функцию `find_best_split`, которая принимает
# те же аргументы, что и функции, реализованные в заданиях выше, в теле функции произвести разбиения,
# начиная с 10/90 и заканчивая 90/10 с шагом 10. При каждом разбиении оценить качество на валидационной выборке
# и выборке `x_test`. Функция должна возвращать датафрейм, с 3-мя столбцами: размер валидационной выборки,
# оценка качества модели на валидационной выборке, оценка качества модели на тестовой выборке.
# Пропуски заполнить медианой, для масштабирования использовать `MinMaxScaler()`.
# Разбение выборок производить с `random_state = 1, shuffle=True`. Проанализировать поведение модели при разных разбиения.
def find_best_split(x, y, x_test, y_test):
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


# Задание 9
# Какое разбиение является оптимальным? Оптимальное разбиение - то разбиение, при котором разница в метрике качества
# между валидационной и тестовой выборкой минимально, и значение метрики на тесте не меньше 76.
# Написать функцию `choose_best_split`, которая принимает результаты расчетов в задании 7 и возвращает размер
# валидационной выборки для наилучшего разбиения
def choose_best_split(scores):
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
