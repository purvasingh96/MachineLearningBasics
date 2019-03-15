#  What pickle does is that it “serialises” the object first before writing it to file.
#  Pickling is a way to convert a python object (list, dict, etc.) into a character stream.

#In our use case- we save training data to avoid training it again and again
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import quandl, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

style.use('ggplot')

data_frame = quandl.get('WIKI/GOOGL')

data_frame=data_frame[['Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']]

data_frame['HL_PCT']=(data_frame['Adj. High'] - data_frame['Adj. Close'])/data_frame['Adj. Close'] * 100.0
data_frame['PCT_Change']=(data_frame['Adj. Close'] - data_frame['Adj. Open'])/data_frame['Adj. Open'] * 100.0

data_frame = data_frame[['Adj. Close', 'HL_PCT', 'Adj. Volume']]

forecast_col = 'Adj. Close'
#last 10 days data to predict future outcome
forecast_out = int(math.ceil(0.01*len(data_frame)))

data_frame['label']=data_frame[forecast_col].shift(-forecast_out)
#dropna will drop missing values
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html


#features - X , labels - y
#X is every col except label

# >>> my_list = [1, 2, 3, 4, 5]
# >>> my_list[:-2]
# [1, 2, 3]

X = np.array(data_frame.drop(['label'], 1))
#scaling data w.r.t training data as well
X = preprocessing.scale(X)
X = X[:-forecast_out]
#future data - Linear Regression implies line of format y=mx+b
#x in y=mx+b is X_lately
#using x, we calculate future y values
X_lately = X[-forecast_out:]
data_frame.dropna(inplace=True)
y = np.array(data_frame['label'])

#cross_validation is deprecated, use model_validation
#separates training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
#fit - train,  score - test
clf.fit(X_train, y_train)
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in =open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
#accuracy is squared error
accuracyLR = clf.score(X_test, y_test)

clfSVM = svm.SVR()
#fit - train,  score - test
clfSVM.fit(X_train, y_train)
#accuracy is squared error

#print(data_frame.head())
accuracySVM = clfSVM.score(X_test, y_test)

print("accuracyLR - ",accuracyLR)

print("accuracySVM - ",accuracySVM)

forecast_set = clf.predict(X_lately)

print("Forecast value - ", forecast_set)

data_frame['Forecast'] = np.nan

last_date = data_frame.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix+one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=86400
    data_frame.loc[next_date]= [np.nan for _ in range(len(data_frame.columns)-1)]+[i]

data_frame['Adj. Close'].plot()
data_frame['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()