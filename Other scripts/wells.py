# to install modules use: py -m pip install module
# import pandas lib as pd
import pandas as pd

# default working directory is C:\Users\:username:
 
# read by default 1st sheet of an excel file
df1 = pd.read_excel('wells.xlsx')
 
print(df1)
print('\n')

# records to be selected
list1 = [1, 2, 3, 7, 8, 9]
 
df2 = df1[df1.FID.isin(list1)]
#print(df2)

# selecting records using query
df3 = df1.query('FID in @list1')
#print(df3)

df3 = df1.query('FID not in @list1')
#print(df3)

# simulate the like keyword using contains syntax
df3 = df1.query('Well_source.str.contains("GAMA")')
#print(df3)

# using the and operator
df3 = df1.query("(Well_source == 'GAMA') and (Lat >= 128.6)")
#print(df3)

# using the or operator
df3 = df1.query("(Well_source == 'DWR') or (Lat >= 128.6)")
print(df3)

# exporting dataframe to excel
# determining the name of the file
file_name = 'Wells_query.xlsx'
  
# saving the excel
df3.to_excel(file_name)