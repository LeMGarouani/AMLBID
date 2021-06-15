from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from termcolor import colored
import pandas as pd


def load_data(path):
    print("Loading data...", flush=True)
    data = pd.read_csv(path, sep = '[;,]')
                        
    # removing an id column if exists
    if 'id' in data.columns:
        data = data.drop('id', 1)     


    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # one-hot encoding
    X = pd.get_dummies(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # splitting data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return data,X_train,Y_train, X_test,Y_test



def lload_data(path):
    print("Loading data...", flush=True)
    data = pd.read_csv(path)
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return data,X_train,Y_train, X_test,Y_test
    