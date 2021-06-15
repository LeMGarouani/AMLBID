from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier


def generate_imports_code(algorithm):
    """Generate all library import calls.

    Parameters
    ----------
    algorithm: string
        name of the recommended learner

    Returns
    -------
    imports: String
        The Python code that imports all required library used in the current
        optimized pipeline
    """
    
    clf_import= {'AdaBoostClassifier': "from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n",
                 'RandomForestClassifier': "from sklearn.ensemble import RandomForestClassifier\n",
                 'SVC': "from sklearn.svm import  SVC\n",
                 'ExtraTreesClassifier': "from sklearn.ensemble import ExtraTreesClassifier\n",
                 'GradientBoostingClassifier': "from sklearn.ensemble import GradientBoostingClassifier\n",
                 'DecisionTreeClassifier': "from sklearn.tree import DecisionTreeClassifier\n",
                 'LogisticRegression': "from sklearn.linear_model import LogisticRegression\n",
                 'SGDClassifier': "from sklearn.linear_model import SGDClassifier\n"
            }
    
    
    imports_basic="""import numpy as np\nimport pandas as pd\nfrom sklearn.metrics import classification_report\nfrom sklearn.model_selection import train_test_split\n"""
    imports=imports_basic+clf_import[algorithm]
    return imports


def generate_pipeline_code(pipeline,DS_path):
        """Generate recommended pipeline code.

    Parameters
    ----------
    pipeline: tuple
        name and recommended pipeline configuration

    Returns
    -------
    code: String
        The Python code recommended pipeline
    """
        
        code= """
# NOTE: Make sure that the target column is labeled 'class' in the data file
data = pd.read_csv('{}')

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model= {}
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
score = model.score(X_test, Y_test)

print(classification_report(Y_test, Y_pred))
print(' Pipeline test accuracy:  %.3f' % score)

                 """.format(DS_path, pipeline)
        return code

def get_pipeline(algorithm,config):
    
    algorithms = {'AdaBoostClassifier': AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
              'RandomForestClassifier': RandomForestClassifier(),
              'SVC': SVC(),
              'ExtraTreesClassifier': ExtraTreesClassifier(),
              'GradientBoostingClassifier': GradientBoostingClassifier(),
              'DecisionTreeClassifier': DecisionTreeClassifier(),
              'LogisticRegression': LogisticRegression(),
              'SGDClassifier': SGDClassifier()}
    
    model=algorithms[algorithm]
    for k, v in config.items():
            model.set_params(**{k: v})
        #print(model)
    return model

def generate_pipeline_file(algorithm,config,DS_path):
    model_conf=get_pipeline(algorithm,config)
    imports=generate_imports_code(algorithm)
    code=generate_pipeline_code(model_conf,DS_path)
    All=imports+code
    filename = 'Recommended_config_implementation' +'.py'
    # open the file to be written
    fo = open(filename, 'w')
    fo.write('%s' % All)
    fo.close()
    return All
    