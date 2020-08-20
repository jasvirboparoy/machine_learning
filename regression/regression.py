import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Get stock prices for google
df = quandl.get('WIKI/GOOGL')
# Get the features we actually care about
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# Calculate the percentage change between the high and the close price
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
# Calculate the percentage change between the open and close price
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Adjust df to only show new features we care about
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# This varible can be changed easily
forecast_col = 'Adj. Close'

# Fill out columns with no data in ML and it will be an outlier
df.fillna(-99999, inplace=True)

# This is 0.01 of df into the future
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

# Assign label column next to data for forecast
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)
df.dropna(inplace=True) 
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)