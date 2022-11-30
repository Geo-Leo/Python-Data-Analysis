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
inc = df['2022 Median Household Income'] / 1000
age = df['2022 Median Age']
time_police = df['Minutes to Police']
time_firedept = df['Minutes to Fire Department']
time_freeway = df['Minutes to Freeway']

df2 = df.assign(Crime_Index = crime, Population = pop, HH_Income = inc, Age = age, Travel_Time_Police = time_police, Travel_Time_Firedept = time_firedept, Travel_Time_Freeway = time_freeway)


expr = "Crime_Index ~ Population + HH_Income + Age + Travel_Time_Police + Travel_Time_Firedept + Travel_Time_Freeway"

#Set up the X and y matrices
y_mat, x_mat = dmatrices(expr, df2, return_type='dataframe')

#Using the statsmodels GLM class
model = sm.GLM(y_mat, x_mat, family=sm.families.Poisson()).fit()

#Print the summary
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(11.9, 6))
plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}, fontproperties = 'monospace') 
plt.axis('off')
#plt.tight_layout()
plt.savefig('output.png')
#print(model.summary())


residual = df['Deviance Residual']
dev = pandas.cut(residual, bins=[-10, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 10], labels=['< -2.5', '-2.5 - -1.5', '-1.5 - -0.5', '-0.5 - 0.5','0.5 - 1.5', '1.5 - 2.5', '> 2.5'])
# adding dev variable to dataframe
df2 = df.assign(Deviance = dev)
#print(df2.to_string())

# adding income / 1000
income = inc
df2 = df2.assign(Income = income)
df2.drop(['2022 Median Household Income'],axis=1, inplace=True)
df2 = df2.rename({"Income": '2022 Median Household Income'}, axis='columns')
import seaborn as sns
import matplotlib.pyplot as plt
#print(df2.to_string())





'''
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
'''




# Plot of Residulas vs. Y^
expr = "Crime_Index ~ Population + HH_Income + Age + Travel_Time_Police + Travel_Time_Firedept + Travel_Time_Freeway"

import numpy
# Poisson Model
crime_log = 7.696369 + -0.000554 * pop + -0.000003 * inc + -0.042893 * age + -0.294963 * time_police + -0.220129 * time_firedept + 0.182358 * time_freeway 
crime_est = round(numpy.exp(crime_log), 0)
df3 = df2.assign(Crime_Index_Estimate = crime_est)
df3 = df3.rename({'Crime_Index_Estimate': 'Crime Index Estimate'}, axis='columns')
summary(crime_est)
#print(df3.to_string())


sns.lmplot(data=df3, y='Deviance Residual', x='Crime Index Estimate', hue='Deviance', palette='coolwarm', fit_reg=False)
#ax = plt.gca()
plt.subplots_adjust(top=0.93)
plt.title("Deviance Residual VS. Predicted")

plt.savefig('Deviance vs predicted.png')


            
# histogram of Residuals
plt.clf() 
import matplotlib.pyplot as plt
Residuals = df2['Deviance Residual']
plt.hist(Residuals, edgecolor='red', bins=8)
#plt.title("Distribution of Deviance Residual")
plt.xlabel("Deviance Residual")
plt.ylabel("Count")

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats as stats

data = np.random.normal(-0.162, 3.447, 250)
mu, std = norm.fit(data) 

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = 68.8 * norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.legend(["Normal\nDistribution"], loc = "upper left")
#plt.tight_layout()
plt.subplots_adjust(top=0.89)
plt.suptitle("Distribution of Deviance Residual")
plt.title("mean = -0.162")

plt.savefig('Histogram residuals.png')



'''
df2.drop(['OBJECTID', 'SOURCE_ID', 'Raw Predicted (CRMCYPROC)', 'Predicted (CRMCYPROC)', 'Deviance Residual', 'Shape__Area', 'Shape__Length'],axis=1, inplace=True)
df3 = df2.drop(['Minutes to Police', 'Minutes to Fire Department', 'Minutes to Freeway'],axis=1, inplace=False)

facet = sns.pairplot(df3)
facet.fig.subplots_adjust(top=0.93) # adjust the Figure in rp
facet.fig.suptitle('Relationships between Variables', font=16)

plt.savefig('Scatterplot matrix.png')
'''