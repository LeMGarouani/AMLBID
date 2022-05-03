#from AMLBID.loader import *
#from MetafeatureExtraction.metafeatures import *



from .recommender_components import loader,MetafeaturesExtractor
from sklearn.model_selection import train_test_split
import pandas as pd

    
def recommendd(ds_path):
    print("Loading data...", flush=True)
    data=pd.read_csv(ds_path,sep=',')
    print("Extracting Meta-features...")
    MF=MetafeaturesExtractor.ExtractMetaFeatures(ds_path).drop(['dataset'], axis=1)
    print("Recommendations generated...")
    Knn=loader.get_neighbors(MF)
    pipeline,p=loader.get_pipelines(Knn[0][0])
    code=generate_pipeline_file(pipeline[0],pipeline[1],ds_path)
    print(colored('Recommended configuration implementation:\n', 'blue', attrs=['bold','underline']))
    print(code)
    return pipeline[1]
  
def recommend(data, metric="", mode=""):
        print("Characterizing data ...")
        MF=MetafeaturesExtractor.extract_data_metafeatures(data)
        print("Building recommendations...")
        Knn=loader.get_neighbors(MF)
        pipelines_list=loader.get_pipelines(Knn[0][0])
        
        NovaNN=loader.get_ANOVA_neighbors(MF)
        pipelines_list[0].append(NovaNN)
        
        if mode=="Recommender":
            pipeline=pipelines_list[0][0]
            model=pipeline[1]
            print("Exporting implementation...")
            generate_pipeline_file(pipeline[0],pipeline[1])
            return model
        
        if mode=="Recommender_Explainer":
            pipeline=pipelines_list[0][0]
            #p=pipelines_list[0][1]
            #acc=pipelines_list[0][2]
            model=pipeline[1]
            a=[pipelines_list,data]
            return model,a

        

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
    
    clf_import= {'AdaBoost': "from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n",
                 'RandomForestClassifier': "from sklearn.ensemble import RandomForestClassifier\n",
                 'SVM': "from sklearn.svm import  SVC\n",
                 'ExtraTrees': "from sklearn.ensemble import ExtraTreesClassifier\n",
                 'GradientBoosting': "from sklearn.ensemble import GradientBoostingClassifier\n",
                 'DecisionTree': "from sklearn.tree import DecisionTreeClassifier\n",
                 'LogisticRegression': "from sklearn.linear_model import LogisticRegression\n",
                 'SGDClassifier': "from sklearn.linear_model import SGDClassifier\n"
            }
    
    
    imports_basic="""import numpy as np\nimport pandas as pd\nfrom sklearn.metrics import classification_report\nfrom sklearn.model_selection import train_test_split\n"""
    imports=imports_basic+clf_import[algorithm]
    return imports


def generate_pipeline_code(pipeline):
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
data = pd.read_csv('Dataset path')

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model= {}
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
score = model.score(X_test, Y_test)

print(classification_report(Y_test, Y_pred))
print(' Pipeline test accuracy:  %.3f' % score)

                 """.format(pipeline)
        return code
    
def generate_pipeline_file(algorithm,pipeline,DS_path=""):
    
    imports=generate_imports_code(algorithm)
    code=generate_pipeline_code(pipeline)
    All=imports+code
    filename = 'Recommended_pypeline' +'.py'
    # open the file to be written
    fo = open(filename, 'w')
    fo.write('%s' % All)
    fo.close()
    return All
    