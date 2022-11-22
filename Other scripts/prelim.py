 read a csv file
import pandas
# header row must not already be in string format
df = pandas.read_csv('SP.csv')
print(df.head())

# select rows using index label
rows = df.loc[1:4]
print(rows)

# select one column
units = df['Units']
print(units)

# remove one column
df.pop('Comments')

# filter using boolean values
filter = df['Completion'] == 'Yes'

# observations that don't satisfy criteria are set to NaN
df2 = df.where(filter)
#print(df2)

# remove NaN values
df3 = df2.dropna()
units = df3['Units']
id = df3['ID']

# check type of object
print(type(units))

# print entire dataset
print(df3.to_string())

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

summary(units)

# print mean and standard deviation
def meanstd(var1):
  import statistics
  mean_var1 = statistics.mean(var1)
  std_var1 = statistics.stdev(var1)
  name = var1.name
  print()
  print(f'Mean and Std for {name}')
  print(f'Mean: {mean_var1:.3f}')
  print(f'Std: {std_var1:.3f}')

meanstd(units)

# histogram with default blue color
import matplotlib.pyplot as plt
'''plt.hist(units)
plt.title("Histogram for verified units")
plt.xlabel("# of units")
plt.ylabel("Frequency")'''
#plt.show()

# two scatterplots on same graph
# using plot.scatter() drops NaN values
# units <= 8
filter2 = df['Units'] > 8
graphdata = df.where(filter2)
graphdata2 = df.where(df3['Units'] <= 8)
plt.scatter(graphdata2['ID'], graphdata2['Units'])

print(filter2)

# scatterplot with ruby red color
# units > 8
plt.scatter(graphdata['ID'], graphdata['Units'], color='#9B111E')
plt.title("Units over time")
plt.xlabel("Date")
plt.ylabel("Units")
# add a legend  
plt.legend(["<= 8 units", "> 8 units"], loc = "upper right")
#plt.show()

# recode ID variable to integer
# ignore warning
df3['ID']= df['ID'].astype(int)
print(df3.head())

# save as csv with no index
'''in order to properly write file, plt.show() must not be 
 ran also or program will continue running as graphs are output'''
df3.to_csv('file2.csv', index=False)