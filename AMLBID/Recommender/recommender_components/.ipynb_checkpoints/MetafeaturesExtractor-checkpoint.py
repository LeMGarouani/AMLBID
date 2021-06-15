import numpy as np
import pandas as pd
import glob
import warnings
import scipy
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mutual_info_score,accuracy_score



def numeric_impute(data, num_cols, method):
    
    num_data = data[num_cols]
    if method == 'mode':
        output = num_data.fillna(getattr(num_data, method)().iloc[0])
    else:
        output = num_data.fillna(getattr(num_data, method)())
    return output

def dict_merge(*args):
    imp = {}
    for dictt in args:
        imp.update(dictt)
    return imp


def summary_stats(data, include_quantiles = False):
    quantiles = np.quantile(data,[0, 0.25, 0.75, 1])
    minn = quantiles[0]
    maxx = quantiles[-1]
    q1 = quantiles[1]
    q3 = quantiles[2]
    mean = np.mean(data)
    std = np.std(data)
    
    if include_quantiles:
        return minn, q1, mean, std, q3, maxx
    else:
        return minn, mean, std, maxx

def pair_corr(data):    
    cors = abs(data.corr().values)
    cors = np.triu(cors,1).flatten()
    cors = cors[cors != 0]
    return cors

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def MI(X, y):
    bins = 10
    
    # check if X and y have the same length
    n = X.shape[1]
    matMI = np.zeros(n)

    for ix in np.arange(n):
        matMI[ix] = calc_MI(X.iloc[:,ix], y, bins)
    
    return matMI

def preprocessing(data):
    
    X = data.iloc[:, :-1]

    # selecting the response variable
    y = data.iloc[:, -1]

    # one-hot encoding
    X = pd.get_dummies(X)

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y
    
def meta_features(data, num_cols, categorical_cols):
    
    metafeatures = {}
    
    target_variable = data.iloc[:, -1]
    
    nr_classes = target_variable.nunique()
    metafeatures['nr_classes'] = nr_classes
    
    nr_instances = data.shape[0]
    metafeatures['nr_instances'] = nr_instances
    
    log_nr_instances = np.log(nr_instances)
    metafeatures['log_nr_instances'] = log_nr_instances
    
    nr_features = data.shape[1]
    metafeatures['nr_features'] = nr_features
    
    log_nr_features = np.log(nr_features)
    metafeatures['log_nr_features'] = log_nr_features
    
    missing_val = data.isnull().sum().sum() + data.isna().sum().sum()
    #metafeatures['missing_val'] = missing_val
    
    # Ratio of Missing Values 
    ratio_missing_val = missing_val / data.size
    #  metafeatures['ratio_missing_val'] = ratio_missing_val
    
    # Number of Numerical Features 
    nr_numerical_features = len(num_cols)
    #  metafeatures['nr_numerical_features'] = nr_numerical_features
    
    # Number of Categorical Features 
    nr_categorical_features = len(categorical_cols)
    #metafeatures['nr_categorical_features'] = nr_categorical_features
    
    # print(data[num_cols].nunique() / data[num_cols].count()) 
    
    
    # Ratio of Categorical to Numerical Features 
    if nr_numerical_features != 0:
        ratio_num_cat = nr_categorical_features / nr_numerical_features
    else:
        ratio_num_cat = 'NaN'
    
    #  metafeatures['ratio_num_cat'] = ratio_num_cat
        
    # Dataset Ratio
    dataset_ratio = nr_features / nr_instances
    metafeatures['dataset_ratio'] = dataset_ratio
        
    # Categorical Features Statistics
    if nr_categorical_features != 0:

        labels = data[categorical_cols].nunique()

        # Labels Sum 
        labels_sum = np.sum(labels)
        
        # Labels Mean 
        labels_mean = np.mean(labels)

        # Labels Std 
        labels_std = np.std(labels)
        
    else:
        labels_sum = 0
        labels_mean = 0
        labels_std = 0
        
    #  metafeatures['labels_sum'] = labels_sum
    #metafeatures['labels_mean'] = labels_mean
    #metafeatures['labels_std'] = labels_std

    return metafeatures



def meta_features2(data, num_cols):
    
    metafeatures = {}
     
    nr_numerical_features = len(num_cols)
    
    if nr_numerical_features != 0:
        
        skewness_values = abs(data[num_cols].skew())
        kurtosis_values = data[num_cols].kurtosis()        
                
        skew_min, skew_q1, \
        skew_mean, skew_std, \
        skew_q3, skew_max = summary_stats(skewness_values, 
                                          include_quantiles=True)
        
        kurtosis_min, kurtosis_q1, \
        kurtosis_mean, kurtosis_std, \
        kurtosis_q3, kurtosis_max = summary_stats(kurtosis_values,
                                                  include_quantiles=True)
        
               
        pairwise_correlations = pair_corr(data[num_cols])
                
        try:
            rho_min, rho_mean, \
            rho_std, rho_max = summary_stats(pairwise_correlations)
        except IndexError:
            pass
                    
    var_names = ['skew_min', 'skew_std', 'skew_mean','skew_q1', 'skew_q3', 'skew_max',
                 'kurtosis_min', 'kurtosis_std','kurtosis_mean', 'kurtosis_q1','kurtosis_q3', 'kurtosis_max',
                 'rho_min', 'rho_max', 'rho_mean','rho_std']

    for var in var_names:
        try:
            metafeatures[var] = eval(var)            
        except NameError:           
            metafeatures[var] = 0
            

    return metafeatures

def shan_entropy(c):
    c_normalized = c[np.nonzero(c)[0]]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def norm_entropy(X):
    bins = 10
    nr_features = X.shape[1]
    n = X.shape[0]
    H = np.zeros(nr_features)
    for i in range(nr_features):
        x = X.iloc[:,i]
        cont = len(np.unique(x)) > bins
        if cont:
            # discretizing cont features 
            x_discr = np.histogram(x, bins)[0]
            x_norm = x_discr / float(np.sum(x_discr))
            H_x = shan_entropy(x_norm)
            
        else:
            x_norm = x.value_counts().values / n 
            H_x = shan_entropy(x_norm)            
        H[i] = H_x
    H /= np.log2(n)
    return H

def meta_features_info_theoretic(X, y):
    
    metafeatures = {}
    nr_instances = X.shape[0]
    
    # Class Entropy
    class_probs = np.bincount(y) / nr_instances 
    class_entropy = shan_entropy(class_probs)
    metafeatures['class_entropy'] = class_entropy

    # Class probability    
    metafeatures['prob_min'], \
    metafeatures['prob_mean'], \
    metafeatures['prob_std'], \
    metafeatures['prob_max'] = summary_stats(class_probs)

    # Norm. attribute entropy
    H = norm_entropy(X) 
    metafeatures['norm_entropy_min'], \
    metafeatures['norm_entropy_mean'], \
    metafeatures['norm_entropy_std'], \
    metafeatures['norm_entropy_max'] = summary_stats(H)
    
    # Mutual information
    mutual_information = MI(X, y)
    metafeatures['mi_min'], \
    metafeatures['mi_mean'], \
    metafeatures['mi_std'], \
    metafeatures['mi_max'] = summary_stats(mutual_information)
    
    # Equiv. nr. of features
    metafeatures['equiv_nr_feat'] = metafeatures['class_entropy'] / metafeatures['mi_mean']
    
    # Noise-signal ratio
    noise = metafeatures['norm_entropy_mean'] - metafeatures['mi_mean']
    metafeatures['noise_signal_ratio'] = noise / metafeatures['mi_mean']
    
    return metafeatures

class LandmarkerModel:
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def accuracy(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        CV_accuracy = accuracy_score(self.y_test, predictions)          
        return CV_accuracy
    
def meta_features_landmarkers(X, y):
    
    metafeatures = {}
    
    k = 10

    kf = StratifiedKFold(n_splits=k, shuffle=True)
    
    model_1nn = KNeighborsClassifier(n_neighbors=1)
    model_dt = DecisionTreeClassifier()
    model_gnb = GaussianNB()
    model_lda = LinearDiscriminantAnalysis()
    
    
    CV_accuracy_1nn = 0
    CV_accuracy_dt = 0
    CV_accuracy_gnb = 0
    CV_accuracy_lda = 0
    
    for train_index, test_index in kf.split(X, y):
        
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            
        y_train, y_test = y[train_index], y[test_index]

         
        CV_accuracy_1nn += LandmarkerModel(model_1nn, X_train, y_train, X_test, y_test).accuracy()
        CV_accuracy_dt += LandmarkerModel(model_dt, X_train, y_train, X_test, y_test).accuracy()
        CV_accuracy_gnb += LandmarkerModel(model_gnb, X_train, y_train, X_test, y_test).accuracy()
        
        try:
            CV_accuracy_lda += LandmarkerModel(model_lda, X_train, y_train, X_test, y_test).accuracy()             
        except scipy.linalg.LinAlgError:
            pass    
    
    CV_accuracy_1nn /= k
    CV_accuracy_dt /= k
    CV_accuracy_gnb /= k
    CV_accuracy_lda /= k
    
    metafeatures['Landmarker_1NN'] = CV_accuracy_1nn
    metafeatures['Landmarker_dt'] = CV_accuracy_dt
    metafeatures['Landmarker_gnb'] = CV_accuracy_gnb
    metafeatures['Landmarker_lda'] = CV_accuracy_lda
        
    return metafeatures

def all_metafeatures(data, num_cols, metafeatures1):

    metafeatures2 = meta_features2(data, num_cols)
    X, y = preprocessing(data)                             
    metafeatures3 = meta_features_info_theoretic(X, y)
    metafeatures4 = meta_features_landmarkers(X, y)  

    metafeatures = dict_merge(metafeatures1, metafeatures2,
                              metafeatures3, metafeatures4)
    return metafeatures

def extract_metafeatures(file):
  
    warnings.filterwarnings("ignore")
    
    
    data = pd.read_csv(file,
                        index_col=None,
                        header=0,
                        sep = '[;,]',
                        na_values='?')

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
    
    data[categorical_cols] = categ_data.fillna(categ_data.mode().iloc[0])
    
       
    metafeatures1 = meta_features(data, num_cols, categorical_cols)
    
    ### Numerical Features Statistics  
    #missing_val = metafeatures1['missing_val'] 
    missing_val=0
        
    if missing_val != 0:
        
        imputation_types = ['mean', 'median', 'mode']

        imputed_data = data.copy()
    
        results = pd.DataFrame()
        for index, num_imput_type in enumerate(imputation_types):
                       
            num_cols = list(num_cols) 
            imputed_data[num_cols] = numeric_impute(data, num_cols, num_imput_type)
            #metafeatures1['num_imput_type'] = num_imput_type
            metafeatures = all_metafeatures(imputed_data, num_cols, metafeatures1)                     
            
            df = pd.DataFrame([metafeatures])
            results = pd.concat([results, df], axis=0)
    else:
        
        #metafeatures1['num_imput_type'] = 0
        metafeatures = all_metafeatures(data, num_cols, metafeatures1)
        
        results = pd.DataFrame([metafeatures])
    
    dataset_name = file.split('\\')[-1]
    
    results['dataset'] = dataset_name
    
    return  results


def extract_data_metafeatures(file):
  
    warnings.filterwarnings("ignore")
        
    if isinstance(file,pd.DataFrame):
        data=file
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
    
    data[categorical_cols] = categ_data.fillna(categ_data.mode().iloc[0])
    
       
    metafeatures1 = meta_features(data, num_cols, categorical_cols)
    
    ### Numerical Features Statistics  
    #missing_val = metafeatures1['missing_val'] 
    missing_val=0
        
    if missing_val != 0:
        
        imputation_types = ['mean', 'median', 'mode']

        imputed_data = data.copy()
    
        results = pd.DataFrame()
        for index, num_imput_type in enumerate(imputation_types):
                       
            num_cols = list(num_cols) 
            imputed_data[num_cols] = numeric_impute(data, num_cols, num_imput_type)
            #metafeatures1['num_imput_type'] = num_imput_type
            metafeatures = all_metafeatures(imputed_data, num_cols, metafeatures1)                     
            
            df = pd.DataFrame([metafeatures])
            results = pd.concat([results, df], axis=0)
    else:
        
        #metafeatures1['num_imput_type'] = 0
        metafeatures = all_metafeatures(data, num_cols, metafeatures1)
        
        results = pd.DataFrame([metafeatures])
    
    #dataset_name = file.split('\\')[-1]
    
    #results['dataset'] = dataset_name
    
    return  results



def ExtractMetaFeatures(path):
 
    allFiles = glob.glob(path +"*.csv")
    #print(allFiles)
    results = pd.DataFrame()
    for idx, file in enumerate(allFiles):
        d_name = file.split('//')[-1]
        print('Dataset {}({})'.format(idx + 1, d_name))
        results = pd.concat([results, extract_metafeatures(file)], axis=0)
        results.head()
    results.to_csv('metafeatures.csv',
                   header=True,
                   index=False)
    return results


