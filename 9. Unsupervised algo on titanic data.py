import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing

df = pd.read_excel('titanic.xls')
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

clf = KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
    predicted_value = np.array(X[i]).astype(float)
    predicted_value = predicted_value.reshape(-1, len(predicted_value))
    prediction = clf.predict(predicted_value)
    print("prediction :: ", prediction)
    if prediction[0]==y[i]:
        correct+=1

print(correct/len(X))

