pip install sklearn-contrib-py-earth

import pandas as pd
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import make_regression
from pyearth import Earth

#create fake regression data
X, y = make_regression(n_samples=5000, n_features=15, n_informative=10, noise=0.5, random_state=5)

# define the model
model = Earth()

# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate the model and collect results
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# report performance
mean(n_scores)