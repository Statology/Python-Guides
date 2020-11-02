#LOAD NECESSARY LIBRARIES
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#LOAD AND VIEW IRIS DATASET
iris = datasets.load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                 columns = iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']
print(df.head())
len(df.index)

#DEFINE PREDICTOR AND RESPONSE VARIABLES
X = df[['s_length', 's_width', 'p_length', 'p_width']]
y = df['species']

#FIT LDA MODEL
model = QuadraticDiscriminantAnalysis()
model.fit(X, y)

#DEFINE METHOD TO EVALUATE MODEL
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#EVALUATE MODEL
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores))

#USE MODEL TO MAKE PREDICTION ON NEW OBSERVATION
new = [5, 3, 1, .4]
model.predict([new])