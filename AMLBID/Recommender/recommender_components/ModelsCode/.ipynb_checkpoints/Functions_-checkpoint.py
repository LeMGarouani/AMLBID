import numpy as np
import pandas as pd
import glob
import time
import warnings
from sys import stdout
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score,\
    recall_score, precision_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,\
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

N = 500

params_decision_tree = {'max_features': np.random.uniform(0.1, 0.9, N),
                        'min_samples_leaf': np.random.choice(np.arange(1, 21, 1), N),
                        'min_samples_split': np.random.choice(np.arange(2, 21, 1), N),
                        'criterion': np.random.choice(['entropy', 'gini'], N)}

params_random_forest = {'n_estimators': np.repeat(100, N),
                        'bootstrap': np.random.choice([True, False], N),
                        'max_features': np.random.uniform(0.1, 0.9, N),
                        'min_samples_leaf': np.random.choice(np.arange(1, 21, 1), N),
                        'min_samples_split': np.random.choice(np.arange(2, 21, 1), N),
                        'criterion': np.random.choice(['entropy', 'gini'], N)}

params_adaboost = {'base_estimator__max_depth': np.random.choice(np.arange(1, 11, 1), N),
                   'algorithm': np.random.choice(['SAMME', 'SAMME.R'], N),
                   'n_estimators': np.random.choice(np.arange(50, 501, 1), N),  # iterations in the paper
                   'learning_rate': np.random.uniform(0.01, 2, N)}

params_svm = {'kernel': np.random.choice(['rbf', 'sigmoid'], N),
              'C': np.random.uniform(2 ** (-5), 2 ** 15, N),
              'coef0': np.random.uniform(-1, 1, N),
              'gamma': np.random.uniform(2 ** (-15), 2 ** 3, N),
              'shrinking': np.random.choice([True, False], N),
              'tol': np.random.uniform(10 ** (-5), 10 ** (-1), N)}

params_gboosting = {'learning_rate': np.random.uniform(0.01, 1, N),
                    'criterion': np.random.choice(['friedman_mse', 'mse'], N),
                   'n_estimators': np.random.choice(np.arange(50, 501, 1), N),
                   'max_depth': np.random.choice(np.arange(1, 11, 1), N),
                   'min_samples_split': np.random.choice(np.arange(2, 21, 1), N),
                   'min_samples_leaf': np.random.choice(np.arange(1, 21, 1), N),
                   'max_features': np.random.uniform(0.1, 0.9, N)}

parameters = {'AdaBoost': params_adaboost,
              'RandomForest': params_random_forest,
              'SVM': params_svm,
              'ExtraTrees': params_random_forest,
              'GradientBoosting': params_gboosting,
              'DecisionTree': params_decision_tree}

models = {'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
          'RandomForest': RandomForestClassifier(),
          'SVM': SVC(),
          'ExtraTrees': ExtraTreesClassifier(),
          'GradientBoosting': GradientBoostingClassifier(),
          'DecisionTree': DecisionTreeClassifier()}


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
            print('{}'.format(num_imput_type))

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
    X = pd.get_dummies(X)

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
    k = 10
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

