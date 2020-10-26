import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt

#create data
df = pd.DataFrame({'hours': [1, 2, 4, 5, 5, 6, 6, 7, 8, 10, 11, 11, 12, 12, 14],
                   'score': [64, 66, 76, 73, 74, 81, 83, 82, 80, 88, 84, 82, 91, 93, 89]})

#create scatterplot
plt.scatter(df.hours, df.score)
plt.title('Hours studied vs. Exam Score')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()

#create boxplot
df.boxplot(column=['score'])

#fit simple linear regression model
y = df['score']
x = df[['hours']]
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())

#produce residual plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'hours', fig=fig)

#produce Q-Q plot
res = model.resid
fig = sm.qqplot(res, fit=True, line="45")
plt.show()