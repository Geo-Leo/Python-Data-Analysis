# Created By  : Geo-Leo
# Created Date: 11/22/22
# version = '5.0'
# ------------------------------------------------------------------
# Preliminary data analysis using Python for GIS project
# ------------------------------------------------------------------

# read a csv file
import pandas
# header row must not already be in string format
df = pandas.read_csv('GeneralizedLinearRegression.csv')

# print 5-number summary
def summary(var1):
  from numpy import percentile
  # calculate Q1, Median, and Q3
  quartiles = percentile(var1, [25, 50, 75])
  # calculate min/max
  data_min, data_max = var1.min(), var1.max()
  # name method is only valid for pandas series object
  # it is the column name
  name = var1.name
  print()
  print(f'Summary of {name}')
  print(f'Min: {data_min:.3f}')
  print(f'Q1: {quartiles[0]:.3f}')
  print(f'Median: {quartiles[1]:.3f}')
  print(f'Q3: {quartiles[2]:.3f}')
  print(f'Max: {data_max:.3f}')

summary(df['Deviance Residual'])


import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

crime = df['2022 Property Crime Index']
pop = df['2022 Total Population']
inc = df['2022 Median Household Income']
age = df['2022 Median Age']
time = df['Travel Time End (Minutes)']
df2 = df.assign(crime = crime, pop = pop, inc = inc, age = age, time = time)


expr = "crime ~ pop + inc + age + time"

#Set up the X and y matrices
y_mat, x_mat = dmatrices(expr, df2, return_type='dataframe')

#Using the statsmodels GLM class, train the Poisson regression model on the training data set.
model = sm.GLM(y_mat, x_mat, family=sm.families.Poisson()).fit()

#Print the training summary.
print(model.summary())







'''

residual = df['Deviance Residual']
dev = pandas.cut(residual, bins=[-10, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 10], labels=['< -2.5', '(-2.5, -1.5)', '(-1.5, -0.5)', '(-0.5, 0.5)','(0.5, 1.5)', '(1.5, 2.5)', '> 2.5'])
# adding dev variable to dataframe
df2 = df.assign(Deviance = dev)

import seaborn as sns
import matplotlib.pyplot as plt


# first Scatterplot
sns.lmplot(data=df2, y='2022 Property Crime Index', x='2022 Total Population', hue='Deviance', palette='coolwarm', fit_reg=False)

import numpy as np

x = df2['2022 Total Population']
y = df2['2022 Property Crime Index']
a, b = np.polyfit(x, y, 1)
#add line of best fit to plot
plt.plot(x, a * x + b)
plt.show()


# second Scatterplot
sns.lmplot(data=df2, y='2022 Property Crime Index', x='2022 Median Household Income', hue='Deviance', palette='coolwarm', fit_reg=False)

import numpy as np

x = df2['2022 Median Household Income']
y = df2['2022 Property Crime Index']
a, b = np.polyfit(x, y, 1)

#add line of best fit to plot
plt.plot(x, a * x + b)
plt.show()


# third Scatterplot
sns.lmplot(data=df2, y='2022 Property Crime Index', x='2022 Median Age', hue='Deviance', palette='coolwarm', fit_reg=False)

import numpy as np

x = df2['2022 Median Age']
y = df2['2022 Property Crime Index']
a, b = np.polyfit(x, y, 1)

#add line of best fit to plot
plt.plot(x, a * x + b)
plt.show()


# Plot of Residulas vs. Y^
pop = df2['2022 Total Population']
inc = df2['2022 Median Household Income']
age = df2['2022 Median Age']
time = df2['Travel Time End (Minutes)']

import numpy

crimelog = 6.65613 + -0.000589 * pop + -0.000003 * inc + -0.018098 * age + -0.290719 * time  # Poisson Model
crimehat = round(numpy.exp(crimelog), 0)
df3 = df2.assign(Crime_Index_Estimate=crimehat)
summary(crimehat)

sns.lmplot(data=df3, y='Deviance Residual', x='Crime_Index_Estimate', fit_reg=False)

import numpy as np

x = df3['Crime_Index_Estimate']
y = df3['Deviance Residual']
a, b = np.polyfit(x, y, 1)

#add line of best fit to plot
plt.plot(x, a * x + b)
plt.show()


# histogram of Residuals
import matplotlib.pyplot as plt

Residuals = df2['Deviance Residual']
plt.hist(Residuals, edgecolor='red', bins=6)
plt.title("Histogram of Residuals")
plt.xlabel("Deviance from estimate")
plt.ylabel("Frequency")
plt.show()
'''