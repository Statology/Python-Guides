#IMPORT NECESSARY PACKAGES
import pandas as pd
from numpy import arange
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold

#LOAD DATA
url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/mtcars.csv"
data_full = pd.read_csv(url)
data = data_full[["mpg", "wt", "drat", "qsec", "hp"]]
data[0:6]

#FIT RIDGE REGRESSION MODEL
#define predictor and response variables
X = data[["mpg", "wt", "drat", "qsec"]]
y = data["hp"]

#define cross-validation method to evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#define model
model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')

#fit model
model.fit(X, y)

#display lambda that produced the lowest test MSE
print(model.alpha_)

#USE MODEL TO PREDICT RESPONSE VALUE OF NEW OBSERVATIONS
#define new observation
new = [24, 2.5, 3.5, 18.5]

#predict hp value using ridge regression model
model.predict([new])