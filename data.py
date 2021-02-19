import pandas as pd
df= pd.read_csv('data.csv')
df.head()
# features of the attributes
df.info()
# number of columns and rows
df.shape
# number of rows per class
df.Class.value_counts()
# bar chart for the classes
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Class' , data = df)
plt.show()
# descriptive statistics
df.describe().transpose()
# Visualizing the data
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
plt.hist(df.Alcohol, density=True, bins=20)  # density=False would make counts
plt.ylabel('Frequency')
plt.xlabel('Alcohol');
#
plt.hist(df['Proline'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Proline');
# 
plt.hist(df['OD280/OD315 of diluted wines'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('OD280/OD315 of diluted wines');
#
plt.hist(df['Hue'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Hue');
#
plt.hist(df['Color intensity'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Color intensity');
#
plt.hist(df['Proanthocyanins'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Proanthocyanins');
#
plt.hist(df['Nonflavanoid phenols'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Nonflavanoid phenols');
#
plt.hist(df['Flavanoids'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Flavanoids');
#
plt.hist(df['Total phenols'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Total phenols');
#
plt.hist(df['Magnesium'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Magnesium');
#
plt.hist(df['Alcalinity of ash'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Alcalinity of ash');
#
plt.hist(df['Ash'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Ash');
#
plt.hist(df['Malic acid'], density=True, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Malic acid');
#