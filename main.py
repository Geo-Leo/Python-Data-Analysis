import pandas
# header row must not already be in string format
df = pandas.read_csv('SP.csv')
print(df.head())

# selecting rows using index label
rows = df.loc[1:4]
print(rows)

# selecting one column
units = df['Units']
print(units)

# filtering using boolean values
filter = df['Completion'] == 'Yes'

# filtering data
# observations that don't satisfy criteria are set to NaN
df2 = df.where(filter)
#print(df2)

# remove NaN values
df3 = df2.dropna()
units = df3['Units']
id = df3['ID']

#print(df3.to_string())


# print 5-number summary
def summary(var1):
  # calculate a 5-number summary
  from numpy import percentile
  quartiles = percentile(var1, [25, 50, 75])
  # calculate min/max
  data_min, data_max = var1.min(), var1.max()
  print('\nSummary of variable')
  print(f'Min: {data_min:.3f}')
  print(f'Q1: {quartiles[0]:.3f}')
  print(f'Median: {quartiles[1]:.3f}')
  print(f'Q3: {quartiles[2]:.3f}')
  print(f'Max: {data_max:.3f}')


summary(units)

def meanstd(var1) :
  import statistics
  mean_units = statistics.mean(var1)
  std_units = statistics.stdev(var1)
  print('\nMean and Std for variable')
  print(f'Mean: {mean_units:.3f}')
  print(f'Std: {std_units:.3f}')

meanstd(units)

import matplotlib.pyplot as plt

plt.hist(units)
plt.title("Histogram for verified units")
plt.xlabel("# of units")
plt.ylabel("Frequency")
plt.show()

plt.scatter(id, units, color='#9B111E')
plt.title("Units over time")
plt.xlabel("Date")
plt.ylabel("Units")
plt.show()

# saving as csv with no index
#df3.to_csv('file1.csv', index=False)
