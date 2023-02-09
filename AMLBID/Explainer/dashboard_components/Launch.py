from AMLBID.Recommender import AMLBID_Recommender
from AMLBID.Explainer import AMLBID_Explainer
from AMLBID.loader import *
import sys

#load dataset
filename = sys.argv[1]
print('AYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', filename, type(filename))
Data,X_train,Y_train,X_test,Y_test=load_data(filename)

#Generate the optimal configurations according to a desired predictive metric
AMLBID,Config=AMLBID_Recommender.recommend(Data, metric="Accuracy", mode="Recommender_Explainer")
AMLBID.fit(X_train, Y_train)

#Generate the interactive explanatory dash
Explainer = AMLBID_Explainer.explain(AMLBID,Config, X_test, Y_test)
Explainer.run()