from sklearn import preprocessing
import pandas as pd
import numpy as np


def prepare_iris_data():
    df = pd.read_csv('iris.data.txt')
    df = handle_non_numerical_data(df)
    y = np.array(df['class'])
    x = np.array(df.drop(['class'], 1))
    return df, x, y


def prepare_breast_cancer_data():
    df = pd.read_csv('breast-cancer.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    y = np.array(df['class'])
    x = np.array(df.drop(['class'], 1))
    return df, x, y


def prepare_titanic_data():
    df = pd.read_excel('titanic.xls')
    df.drop(['body', 'name', 'home.dest'], 1, inplace=True)
    df.infer_objects()
    df.fillna(0, inplace=True)
    df = handle_non_numerical_data(df)
    df.drop(['boat'], 1, inplace=True)
    y = np.array(df['survived'])
    x = np.array(df.drop(['survived'], 1).astype(float))
    x = preprocessing.scale(x)
    return df, x, y


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
    return df
