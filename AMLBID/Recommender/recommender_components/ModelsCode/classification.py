import numpy as np
import pandas as pd
import glob
import time
import warnings
from sys import stdout
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
N = 500



params_decision_tree = {'min_impurity_decrease' : np.random.exponential(scale=0.01, size=N),
                        'max_features' : np.random.choice(list(np.arange(0.01, 1., 0.01)) + ['sqrt', 'log2', None], size=N),
                        'criterion' : np.random.choice(['gini', 'entropy'], size=N),
                        'max_depth' : np.random.choice(list(range(1, 51)) + [None], size=N)}

params_random_forest = {'n_estimators' : np.random.choice(list(range(50, 1001, 50)), size=N),
                        'min_impurity_decrease' : np.random.exponential(scale=0.01, size=N),
                        'max_features' : np.random.choice(list(np.arange(0.01, 1., 0.01)) + ['sqrt', 'log2', None], size=N),
                        'criterion' : np.random.choice(['gini', 'entropy'], size=N),
                        'max_depth' : np.random.choice(list(range(1, 51)) + [None], size=N)}

params_adaboost = {'base_estimator__max_depth': np.random.choice(np.arange(1, 11, 1), N),
                   'n_estimators': np.random.choice(list(range(50, 1001, 50)), N),  # iterations in the paper
                   'learning_rate': np.random.uniform(low=1e-10, high=5., size=N)}

params_svm = {'estimator__base_estimator__C': np.random.uniform(low=1e-10, high=500., size=N),
              'estimator__base_estimator__gamma': np.random.choice(list(np.arange(0.001, 1.01, 0.05)), size=N),
              'estimator__base_estimator__kernel': np.random.choice(['poly', 'rbf'], size=N),
              'estimator__base_estimator__degree': np.random.choice([2, 3], size=N),
              'estimator__base_estimator__coef0': np.random.uniform(low=0., high=10., size=N)}

params_gboosting = {'n_estimators' : np.random.choice(list(range(50, 1001, 50)), size=N),
                    'min_impurity_decrease' : np.random.exponential(scale=0.01, size=N),
                    'max_features' : np.random.choice(list(np.arange(0.01, 1., 0.01)) + ['sqrt', 'log2', None], size=N),
                    'learning_rate' : np.random.uniform(low=1e-10, high=5., size=N),
                    'loss' : np.random.choice(['deviance', 'exponential'], size=N),
                    'max_depth' : np.random.choice(list(range(1, 51)) + [None], size=N)}

params_lr={ 'C' : np.random.uniform(low=1e-10, high=10., size=2*N),
           'penalty' : np.repeat(['l2']+['l1'], N),
           'dual': np.repeat([False]+[True], N),
           'fit_intercept':np.random.choice([True, False], size=2*N)}


param_pass_agg={ 'C' : np.random.uniform(low=1e-10, high=10., size=N),
                 'loss' : np.random.choice(['hinge', 'squared_hinge'], size=N),
                 'fit_intercept' : np.random.choice([True, False], size=N)}

param_sgd={'loss' : np.random.choice(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], size=N),
            'penalty' : np.random.choice(['l2', 'l1', 'elasticnet'], size=N),
            'alpha' :  np.random.exponential(scale=0.01, size=N),
            'learning_rate' : np.random.choice(['constant', 'optimal', 'invscaling'], size=N),
            'fit_intercept' : np.random.choice([True, False], size=N),
            'l1_ratio' : np.random.uniform(low=0., high=1., size=N),
            'eta0' : np.random.uniform(low=0., high=5., size=N),
            'power_t' : np.random.uniform(low=0., high=5., size=N)}

params_linear_svc={'C' : np.random.uniform(low=1e-10, high=10., size=N),
                   'loss' : np.random.choice(['hinge', 'squared_hinge'], size=N),
                   'penalty' : np.random.choice(['l1', 'l2'], size=N),
                   'dual' : np.random.choice([True, False], size=N),
                   'fit_intercept' : np.random.choice([True, False], size=N)}

params_xgb={ 'n_estimators' : np.random.choice(list(range(50, 1001, 50)), size=N),
            'learning_rate' : np.random.uniform(low=1e-10, high=5., size=N),
            'gamma' : np.random.uniform(low=0., high=1., size=N),
            'max_depth' : np.random.choice(list(range(1, 51)) + [None], size=N),
            'subsample' : np.random.uniform(low=0., high=1., size=N)}

parameters = {'AdaBoost': params_adaboost,
              'RandomForest': params_random_forest,
              'SVM': params_svm,
              'ExtraTrees': params_random_forest,
              'GradientBoosting': params_gboosting,
              'DecisionTree': params_decision_tree,
              'LogisticRegression': params_lr,
              'PassiveAggressiveClassifier':param_pass_agg,
              'SGDClassifier' : param_sgd,
              'LinearSVC' : params_linear_svc,
              'XGBClassifier': params_xgb}
n_estimators=100
models = {'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
          'RandomForest': RandomForestClassifier(),
          'SVM': OneVsRestClassifier(BaggingClassifier(SVC(), max_samples=1.0 / n_estimators,n_estimators=100, bootstrap=True)),
          'ExtraTrees': ExtraTreesClassifier(),
          'GradientBoosting': GradientBoostingClassifier(),
          'DecisionTree': DecisionTreeClassifier(),
          'LogisticRegression': LogisticRegression(),
          'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
          'SGDClassifier': SGDClassifier(),
          'LinearSVC' : LinearSVC(),
          'XGBClassifier': XGBClassifier()}


def classification_per_algorithm(path, algorithm):
    """
    Fits different models with random configurations 
    on each of the datasets in path for the specified ML algorithm
    
    Inputs:
            path - (str) directory of the datasets
            algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
    Outputs: 
            writes the results on a csv file
    """
    
    warnings.filterwarnings("ignore")
    all_files = glob.glob(path + '*.csv')
    all_datasets = len(all_files)

    results = pd.DataFrame()
    start_all = time.perf_counter()
    for index, file in enumerate(all_files):
        print('Dataset {}({}) out of {} \n'.format(index + 1, file, all_datasets), flush=True)
        try:
            file_logs = classification_per_dataset(file, algorithm, models, parameters)
            results = pd.concat([results, file_logs], axis=0)

            results.to_csv('{}_results.csv'.format(algorithm),
                           header=True,
                           index=False)
        except Exception as e:
            print('The following error occurred in case of the dataset {}: \n{}'.format(file, e))
    end_all = time.perf_counter()
    time_taken = (end_all - start_all) / 3600
    stdout.write("Performance data is collected! \n ")
    print('Total time: {} hours'.format(time_taken))


def classification_per_dataset(file, algorithm, models, parameters):
    """
    Gathers the performance data for each random configuration 
    on the given dataset for the specified ML algorithm  and 
    performs data imputation if necessary
    
    Inputs:
            file - (str) name of the dataset
            algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
            models - (dict) key: algorithm, 
                            value: the class of the algorithms
            parameters - (dict) key: algorithm
                                value: the configuration space of 
                                       the algorithm
    Outputs: 
            final_logs - (DataFrame) performance data
            
    """
    
    data = pd.read_csv(file,
                       index_col=None,
                       header=0,
                       na_values='?')

    # making the column names lower case
    data.columns = map(str.lower, data.columns)

    # removing an id column if exists
    if 'id' in data.columns:
        data = data.drop('id', 1)

    # remove columns with only NaN values
    empty_cols = ~data.isna().all()
    data = data.loc[:, empty_cols]
    
    # identifying numerical and categorical features
    cols = set(data.columns)
    num_cols = set(data._get_numeric_data().columns)
    categorical_cols = list(cols.difference(num_cols))

    # data imputation for categorical features
    categ_data = data[categorical_cols]
    data[categorical_cols] = categ_data.fillna(categ_data.mode().iloc[0])

    # defining the random configurations
    combinations = get_combinations(parameters, algorithm)

    # data imputation for numeric features
    if data.isna().values.any():

        imputation_types = ['mean', 'median', 'mode']

        final_logs = pd.DataFrame()
        
        imputed_data = data.copy()

        for index, num_imput_type in enumerate(imputation_types):
            #print('{}'.format(num_imput_type))

            imputed_data[list(num_cols)] = numeric_impute(data, num_cols, num_imput_type)

            # logs per imputation method
            logs = get_logs(imputed_data, num_imput_type, algorithm,
                            file, combinations, models)

            final_logs = pd.concat([final_logs, logs], axis=0)

    else:
        num_imput_type = None

        final_logs = get_logs(data, num_imput_type, algorithm,
                              file, combinations, models)

    return final_logs


def get_logs(data, num_imput_type, algorithm, file,
             combinations, models):
    """
    Gathers the performance data for each random configuration 
    on the given dataset for the specified ML algorithm
    
    Inputs:
            data - (DataFrame) dataset, where the last column 
                               contains the response variable 
            num_imput_type - (str or None) imputation type that takes 
                              one of the following values
                              {'mean', 'median', 'mode', None}
            
            algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
            file - (str) name of the dataset
            
            combinations - (DataFrame) contains the random configurations
                            of the given algorithm
            
            models - (dict) key: algorithm, 
                            value: the class of the algorithms
            
    Outputs: 
            logs - (DataFrame) performance data
            
    """
    # excluding the response variable
    X = data.iloc[:, :-1]

    # selecting the response variable
    y = data.iloc[:, -1]

    # one-hot encoding
    #X = pd.get_dummies(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    num_labels = len(np.unique(y))

    # binarizing the labels for some of the metrics
    # in case of more than 2 labels
    if num_labels > 2:
        multilabel = True
        lb = LabelBinarizer()
        lb.fit(y)
        y_sparse = lb.transform(y)
    else:
        multilabel = False

    # scaling the input in case of SVM algorithm
    if algorithm == 'SVM':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        scaled = True
    else:
        scaled = False

    # setting the number of folds of the Cross-Validation
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True)

    logs = combinations.copy()
    n_comb = logs.shape[0]
    logs.insert(loc=0, column='dataset', value=file)
    logs['imputation'] = num_imput_type

    for index in range(n_comb):
        print('{}/{}'.format(index + 1, n_comb))
        
        params = dict(zip(combinations.columns,
                      list(combinations.iloc[index,:])))
        
        model = models[algorithm]
        model.set_params(**params)

        i = 0

        cv_train_time = np.zeros(k)
        cv_test_time = np.zeros(k)
        acc_tr = np.zeros(k)
        accuracy = np.zeros(k)
        f1 = np.zeros(k)
        recall = np.zeros(k)
        precision = np.zeros(k)
        auc = np.zeros(k)

        for train_index, test_index in kf.split(X, y):
            stdout.write("\rCV {}/{}".format(i+1, k))

            if not scaled:
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            else:
                X_train, X_test = X[train_index, :], X[test_index, :]
            
            y_train, y_test = y[train_index], y[test_index]
            try:
                start_tr = time.perf_counter()
                model.fit(X_train, y_train)
                end_tr = time.perf_counter()
                train_time = end_tr - start_tr

                predictions_tr = model.predict(X_train)
                acc_tr[i] = accuracy_score(y_train, predictions_tr)
                start_ts = time.perf_counter()
                predictions = model.predict(X_test)
                end_ts = time.perf_counter()
                test_time = end_ts - start_ts
                cv_train_time[i] = train_time
                cv_test_time[i] = test_time
                accuracy[i] = accuracy_score(y_test, predictions)

                if multilabel:
                    y_true = y_sparse[test_index, :]
                    predictions = lb.transform(predictions)
                else:
                    y_true = y_test

                f1[i], recall[i], precision[i], auc[i] = other_metrics(y_true,
                                                                       predictions,
                                                                       multilabel)
                i += 1

                stdout.flush()
            except KeyboardInterrupt:
                sys.exit(1)
            # This is a catch-all to make sure that the evaluation won't crash due to a bad parameter
            # combination or bad data. Turn this off when debugging!
            except Exception as e:
                continue

        logs.loc[index, "Mean_Train_time"] = np.mean(cv_train_time)
        logs.loc[index, "Std_Train_time"] = np.std(cv_train_time)
        logs.loc[index, "Mean_Test_time"] = np.mean(cv_test_time)
        logs.loc[index, "Std_Test_time"] = np.std(cv_test_time)
        logs.loc[index, 'CV_accuracy_train'] = np.mean(acc_tr)
        logs.loc[index, 'CV_accuracy'] = np.mean(accuracy)
        logs.loc[index, 'CV_f1_score'] = np.mean(f1)
        logs.loc[index, 'CV_recall'] = np.mean(recall)
        logs.loc[index, 'CV_precision'] = np.mean(precision)
        logs.loc[index, 'CV_auc'] = np.mean(auc)
        logs.loc[index, 'Std_accuracy_train'] = np.std(acc_tr)
        logs.loc[index, 'Std_accuracy'] = np.std(accuracy)
        logs.loc[index, 'Std_f1_score'] = np.std(f1)
        logs.loc[index, 'Std_recall'] = np.std(recall)
        logs.loc[index, 'Std_precision'] = np.std(precision)
        logs.loc[index, 'Std_auc'] = np.std(auc)

        print('\n')

    return logs


def get_combinations(parameters, algorithm):
    """
    Creates a DataFrame of the random configurations
    of the given algorithm
    
    Inputs:
            parameters - (dict) key: algorithm
                                value: the configuration space of 
                                       the algorithm
            algorithm - (str) takes one of the following options
                        {RandomForest, AdaBoost, ExtraTrees, 
                         SVM, GradientBoosting}
            
    Outputs: 
            combinations - (DataFrame) realizations of the 
                            random configurations
            
    """
    param_grid = parameters[algorithm]
    combinations = pd.DataFrame(param_grid)
    return combinations


def other_metrics(y_true, predictions, multilabel):
    """
    Treating the case of multiple labels for 
    computing performance measures suitable for 
    binary labels
    
    Inputs:
            y_true - (array or sparse matrix) true values of the 
                      labels
            predictions - (array or sparse matrix) predicted values of the 
                      labels 
            multilabel - (boolean) specifies if there are 
                          multiple labels (True)
    Outputs: 
            f1, recall, precision, auc - (float) performace metrics
            
    """
    
    if multilabel:
        f1 = f1_score(y_true, predictions, average='micro')
        recall = recall_score(y_true, predictions, average='micro')
        precision = precision_score(y_true, predictions, average='micro')
        auc = roc_auc_score(y_true, predictions, average='micro')
    else:
        f1 = f1_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        auc = roc_auc_score(y_true, predictions)

    return f1, recall, precision, auc


def numeric_impute(data, num_cols, method):
    """
    Performs numerical data imputaion based 
    on the given method
    
    Inputs:
            data - (DataFrame) dataset with missing 
                     numeric values 
            num_cols - (set) numeric column names
            method - (str) imputation type that takes 
                              one of the following values
                              {'mean', 'median', 'mode'}
            
    Outputs: 
            output - (DataFrame) dataset with imputed missing values 
            
    """
    num_data = data[list(num_cols)]
    if method == 'mode':
        output = num_data.fillna(getattr(num_data, method)().iloc[0])
    else:
        output = num_data.fillna(getattr(num_data, method)())
    return output

