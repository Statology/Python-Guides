import pandas as pd
from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold

#specify URL where data is located
url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/mtcars.csv"

#read in data
data_full = pd.read_csv(url)

#select subset of data
data = data_full[["mpg", "wt", "drat", "qsec", "hp"]]

#view first six rows of data
data[0:6]

#define predictor and response variables
X = data[["mpg", "wt", "drat", "qsec"]]

y = data["hp"]

#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#define model
model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)

#fit model
model.fit(X, y)

#display lambda that produced the lowest test MSE
print(model.alpha_)

#define new observation
new = [24, 2.5, 3.5, 18.5]

#predict hp value using lasso regression model
model.predict([new])