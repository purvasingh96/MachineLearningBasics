import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd
from sklearn import preprocessing

df = pd.read_excel('titanic.xls')
# making a copy of original titanic.xls
original_df = pd.DataFrame.copy(df)
print(df.head())
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


# step-1 handling non-numeric data
# convert every non-numeric data to numeric data for classification

def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_values = {}
        def convert_to_int(nonInt):
            return text_digit_values[nonInt]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique]=x
                    x+=1

            df[column]= list(map(convert_to_int, df[column]))

    return df

df = handle_non_numeric_data(df)
# print(df.head)

X = np.array(df.drop(['survived'], 1).astype(float))
# scaling would standardize the data
# standardization - 0 mean and Unit variance
# normal distribution of data
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_
print("labels :: ", labels)

# adding new column to original data frame
original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i]=labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    # i - no. of clusters
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[ temp_df['survived']==1]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i]=survival_rate

print("survival rate :: ", survival_rates)


