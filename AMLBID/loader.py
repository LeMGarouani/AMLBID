from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from termcolor import colored
import pandas as pd



def numeric_impute(data, num_cols, imput_type='mean'):
    
    num_data = data[num_cols]
    if imput_type == 'mode':
        output = num_data.fillna(getattr(num_data, imput_type)().iloc[0])
    else:
        output = num_data.fillna(getattr(num_data, imput_type)())
    return output

def cat_data_encode(data, categorical_cols):
    le = LabelEncoder()
    for index, categorical_col in enumerate(categorical_cols):
        data[categorical_col] =le.fit_transform(data[categorical_col])
    return data[categorical_cols]


def preprocess(file, imputation_type="mean"):
    
    data = pd.read_csv(file, sep = '[;,]',engine='python')
    
    print("Preprocessing data...", flush=True)
    data.columns = map(str.lower, data.columns)
    
    # removing an id column if exists
    if 'id' in data.columns:
        data = data.drop('id', 1)
        
    # remove constant columns
    data = data.loc[:, (data != data.iloc[0]).any()]
    const_col = data.std().index[data.std() == 0]
    data = data.drop(const_col,axis=1) 
    
    # remove columns with only NaN values
    empty_cols = ~data.isna().all()
    data = data.loc[:, empty_cols]
    
    cols = set(data.columns)
    num_cols = set(data._get_numeric_data().columns)
    categorical_cols = list(cols.difference(num_cols))

    # data imputation for categorical features
    categ_data = data[categorical_cols]
    if len(categorical_cols) !=0:
        data[categorical_cols] = categ_data.fillna(categ_data.mode().iloc[0])
    
       
    missing_val = data.isnull().sum().sum() + data.isna().sum().sum()
        
    if missing_val != 0:
        
        #imputation_type = "mode"#['median', 'mode','mean' ]

        procecced_data = data.copy()
    
        results = pd.DataFrame()
        
        num_cols = list(num_cols) 
        procecced_data[num_cols] = numeric_impute(data, num_cols, imputation_type)
        procecced_data[categorical_cols] = cat_data_encode(procecced_data, categorical_cols)
        return procecced_data

        
    return data
    

    
def load_data(path):
    print("Loading data...", flush=True)
    #data = pd.read_csv(path, sep = '[;,]')
    data = preprocess(path)                    
    # removing an id column if exists
    
    #if 'id' in data.columns:
    #    data = data.drop('id', 1)     


    

    # one-hot encoding
    #X = pd.get_dummies(X)
    
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
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
   
    
    
def DataOverview (ds):
    #dataset general info
    n_var = ds.shape[1]
    n_obs = ds.shape[0]
    n_missing = ds.isnull().sum().sum() + ds.isna().sum().sum()
    n_classes = ds.iloc[:, -1].nunique()
    dup_rows =len(ds)-len(ds.drop_duplicates())
    #varibales (data type) infos
    Numeric = ds.select_dtypes(include='number').shape[1]
    Categorical = ds.select_dtypes(include='object').shape[1]
    Boolean = ds.select_dtypes(include='bool').shape[1]
    Date = ds.select_dtypes(include='datetime64').shape[1]
    Unsupported = 0
    
    dataInfo = [n_var, n_obs, n_missing, n_classes, dup_rows]
    VarInfo= [Numeric, Categorical, Boolean, Date, Unsupported]
    return dataInfo, VarInfo