from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.base import clone
from termcolor import colored
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import ast
import os


# Load learning metafeatures
def load_MF():
    metafeatures = pd.read_csv(os.path.dirname(__file__)+'/../builtins/KnowledgeBase/metafeatures.csv',sep=',')
    return metafeatures

# Load systems knowledge base
def load_KB():
    knowledgebase=pd.read_csv(os.path.dirname(__file__)+'/../builtins/KnowledgeBase/KB_Acc.csv',sep=';') #best pipelines in KB
    return knowledgebase

# Locate the most similar neighbors
def get_neighbors(DS, num_neighbors=1):
    Data=load_MF()
    MF=Data.drop(['dataset'], axis=1).to_numpy()
    DS=DS.to_numpy()
    distances = []
    for mf in MF:
        dist = np.linalg.norm(mf - DS)
        distances.append((dist))
    ind=np.argsort(distances)[:num_neighbors]
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append([Data.iloc[ind[i],-1]])
        neighbors[i].append(distances[ind[i]])
    return neighbors

# Locate the most similar neighbors
def get_ANOVA_neighbors(DS, num_neighbors=1):
    Data=pd.read_csv(os.path.dirname(__file__)+'/../builtins/KnowledgeBase/MF_FOR_ANOVA.csv',sep=',')
    MF=Data.drop(['dataset'], axis=1).to_numpy()
    DS=DS.to_numpy()
    distances = []
    for mf in MF:
        dist = np.linalg.norm(mf - DS)
        distances.append((dist))
    ind=np.argsort(distances)[:num_neighbors]
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append([Data.iloc[ind[i],-1]])
        neighbors[i].append(distances[ind[i]])
    return neighbors[0][0]


def get_neighbor_data(neighbor):
    bkb=load_KB()
    clf=bkb[(bkb.dataset==neighbor)].classifier.tolist()
    pipe=bkb[(bkb.dataset==neighbor)].parameters.tolist()
    acc=bkb[(bkb.dataset==neighbor)].CV_accuracy.tolist()
    return clf, pipe,acc


def get_pipelines(neighbor):
    pipelines_list=[]
    algorithms = {'AdaBoostClassifier': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                  'RandomForestClassifier': RandomForestClassifier(),
                  'ExtraTreesClassifier': ExtraTreesClassifier(),
                  'GradientBoostingClassifier': GradientBoostingClassifier(),
                  'DecisionTreeClassifier': DecisionTreeClassifier(),
                  'LogisticRegression': LogisticRegression(),
                  'SGDClassifier': SGDClassifier(),
                  'SVM': SVC()}
    
    classifiers,pipelines,accuracies=get_neighbor_data(neighbor)

    for clf,p,acc in zip(classifiers,pipelines,accuracies) :
        a=None
        model=algorithms[clf]
        p=eval(p)
        for k, v in p.items():
            model.set_params(**{k: v})
        a=""+clf+"",clone(model)
        pipelines_list.append([a,p,acc])
    return pipelines_list
