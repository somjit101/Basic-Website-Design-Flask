import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def readBankMarketingDataset():
    # ----------------------- Training Data --------------------------------
    data_train = pd.read_csv(r"./datasets/Bank Marketing/bank-additional-full.csv", na_values=['NA'])

    columns = data_train.columns.values[0].split(';')
    columns = [column.replace('"', '') for column in columns]
    data_train = data_train.values
    data_train = [items[0].split(';') for items in data_train]
    data_train = pd.DataFrame(data_train, columns=columns)

    data_train['job'] = data_train['job'].str.replace('"', '')
    data_train['marital'] = data_train['marital'].str.replace('"', '')
    data_train['education'] = data_train['education'].str.replace('"', '')
    data_train['default'] = data_train['default'].str.replace('"', '')
    data_train['housing'] = data_train['housing'].str.replace('"', '')
    data_train['loan'] = data_train['loan'].str.replace('"', '')
    data_train['contact'] = data_train['contact'].str.replace('"', '')
    data_train['month'] = data_train['month'].str.replace('"', '')
    data_train['day_of_week'] = data_train['day_of_week'].str.replace('"', '')
    data_train['poutcome'] = data_train['poutcome'].str.replace('"', '')
    data_train['y'] = data_train['y'].str.replace('"', '')

    # --------------------------------------------------------------------------

    # --------------------------- Testing Data --------------------------------

    data_test = pd.read_csv(r"datasets/Bank Marketing/bank-additional.csv", na_values=['NA'])
    data_test = data_test.values
    data_test = [items[0].split(';') for items in data_test]
    data_test = pd.DataFrame(data_test, columns=columns)

    data_test['job'] = data_test['job'].str.replace('"', '')
    data_test['marital'] = data_test['marital'].str.replace('"', '')
    data_test['education'] = data_test['education'].str.replace('"', '')
    data_test['default'] = data_test['default'].str.replace('"', '')
    data_test['housing'] = data_test['housing'].str.replace('"', '')
    data_test['loan'] = data_test['loan'].str.replace('"', '')
    data_test['contact'] = data_test['contact'].str.replace('"', '')
    data_test['month'] = data_test['month'].str.replace('"', '')
    data_test['day_of_week'] = data_test['day_of_week'].str.replace('"', '')
    data_test['poutcome'] = data_test['poutcome'].str.replace('"', '')
    data_test['y'] = data_test['y'].str.replace('"', '')

    # -------------------   Preprocessing and Data Cleaning   ----------------------

    data = pd.concat([data_train, data_test])
    data.replace(['basic.6y', 'basic.4y', 'basic.9y'], 'basic', inplace=True)
    data = data[data.job != 'unknown']
    data = data[data.marital != 'unknown']
    data = data[data.loan != 'unknown']
    data = data[data.education != 'illiterate']
    # -------------------------------------------------------------------------------
    return data_train, data_test, data


def categorize(df):
    new_df = df.copy()
    le = preprocessing.LabelEncoder()

    new_df['job'] = le.fit_transform(new_df['job'])
    new_df['marital'] = le.fit_transform(new_df['marital'])
    new_df['education'] = le.fit_transform(new_df['education'])
    new_df['default'] = le.fit_transform(new_df['default'])
    new_df['housing'] = le.fit_transform(new_df['housing'])
    new_df['month'] = le.fit_transform(new_df['month'])
    new_df['loan'] = le.fit_transform(new_df['loan'])
    new_df['contact'] = le.fit_transform(new_df['contact'])
    new_df['day_of_week'] = le.fit_transform(new_df['day_of_week'])
    new_df['poutcome'] = le.fit_transform(new_df['poutcome'])
    new_df['y'] = le.fit_transform(new_df['y'])

    for cols in new_df:
        new_df[cols] = pd.to_numeric(new_df[cols])

    return new_df

def remove_outliers(df, column , minimum, maximum):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values<minimum, col_values>maximum), col_values.mean(), col_values)
    return df
