# read a csv file
import pandas
# header row must not already be in string format
df = pandas.read_csv('GeneralizedLinearRegression.csv')

residual = df['Deviance Residual']
dev = pandas.cut(residual, bins=[-10,-2.5,-1.5,-0.5,0.5,1.5,2.5,10], labels=['-<2.5','(-2.5, -1.5)','(-1.5, -0.5)','(-0.5, 0.5)','(0.5, 1.5)','(1.5, 2.5)','>2.5'])
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
plt.plot(x, a*x+b)
plt.show()


# second Scatterplot
sns.lmplot(data=df2, y='2022 Property Crime Index', x='2022 Median Household Income', hue='Deviance', palette='coolwarm', fit_reg=False)

import numpy as np
x = df2['2022 Median Household Income']
y = df2['2022 Property Crime Index']
a, b = np.polyfit(x, y, 1)

#add line of best fit to plot
plt.plot(x, a*x+b)
plt.show()


# third Scatterplot
sns.lmplot(data=df2, y='2022 Property Crime Index', x='2022 Median Age', hue='Deviance', palette='coolwarm', fit_reg=False)

import numpy as np
x = df2['2022 Median Age']
y = df2['2022 Property Crime Index']
a, b = np.polyfit(x, y, 1)

#add line of best fit to plot
plt.plot(x, a*x+b)
plt.show()


# Plot of Residulas vs. Y^
pop = df2['2022 Total Population']
inc = df2['2022 Median Household Income']
age = df2['2022 Median Age']
time = df2['Travel Time End (Minutes)']

import numpy
crimelog = 6.65613 + -0.000589*pop + -0.000003*inc + -0.018098*age + -0.290719*time  # Poisson Model
crimehat = round(numpy.exp(crimelog), 0)
df3 = df2.assign(Crime_Index_Estimate = crimehat)
summary(crimehat)

sns.lmplot(data=df3, y='Deviance Residual', x='Crime_Index_Estimate', fit_reg=False)

import numpy as np
x = df3['Crime_Index_Estimate']
y = df3['Deviance Residual']
a, b = np.polyfit(x, y, 1)

#add line of best fit to plot
plt.plot(x, a*x+b)
plt.show()


# histogram of Residuals
import matplotlib.pyplot as plt
Residuals = df2['Deviance Residual']
plt.hist(Residuals, edgecolor='red', bins=6)
plt.title("Histogram of Residuals")
plt.xlabel("Deviance from estimate")
plt.ylabel("Frequency")
plt.show()