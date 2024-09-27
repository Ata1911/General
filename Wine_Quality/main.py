import numpy as np
import pandas as pd 
from time import time 
from IPython.display import display  

import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv("C:/Users/Yusuf Ata/data_science/General/Wine_Quality/winequality-red.csv", sep=",")


display(data.head())
#data.isnull().any()

#data.info()

n_wines = data.shape[0]

quality_above_6 = data.loc[(data['quality'] > 6)]
n_above_6 = quality_above_6.shape[0]

quality_below_5 = data.loc[(data['quality'] < 5)]
n_below_5 = quality_below_5.shape[0]

quality_between_5 = data.loc[(data['quality'] >= 5) & (data['quality'] <= 6)]
n_between_5 = quality_between_5.shape[0]

greater_percent = n_above_6*100/n_wines

# Print the results
print("Total number of wine data: {}".format(n_wines))
print("Wines with rating 7 and above: {}".format(n_above_6))
print("Wines with rating less than 5: {}".format(n_below_5))
print("Wines with rating 5 and 6: {}".format(n_between_5))
print("Percentage of wines with quality 7 and above: {:.2f}%".format(greater_percent))

# Some more additional data analysis
display(np.round(data.describe()))

sns.set(style = "whitegrid")

plt.figure(figsize = (8,6))
sns.countplot(data = data, x = 'quality', palette = 'coolwarm')

plt.title("Distribution of Wine Quality Ratings", fontsize=15)
plt.xlabel("Wine Quality", fontsize=12)
plt.ylabel("Count", fontsize=12)

# Show the plot


pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde')

correlation = data.corr()
# display(correlation)
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")

#plt.show()

fixedAcidity_pH = data[['pH', 'fixed acidity']]

gridA = sns.JointGrid(x = "fixed acidity" ,y = "pH" , data = fixedAcidity_pH)

gridA = gridA.plot_joint(sns.regplot, scatter_kws = {"s":10})

gridA = gridA.plot_marginals(sns.distplot)

plt.show()

for feature in data.keys():
    Q1 = np.percentile(data[feature], q = 25)

    Q3 = np.percentile(data[feature] , q = 75)

    interquartile_range = Q3 - Q1
    step = 1.5 * interquartile_range

    print("Data points considered outliners for the feature '{}':".format(feature))
    display(data[~((data[feature] >= Q1 -step) & (data[feature] <= Q3 + step))])

    outliners = []
    
    good_data = data.drop(data.index[outliners]).reset_index(drop = True)