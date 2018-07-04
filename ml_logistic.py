import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import fix_yahoo_finance as yf
yf.pdr_override()
import datetime
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf


def create_dataset(stock_symbol, start_date, end_date, lags=5):

    
	df = pdr.get_data_yahoo(stock_symbol, start_date,end_date)

    
	
	tslag = pd.DataFrame(index=df.index)
	tslag["Today"] = df["Adj Close"]
	tslag["Volume"] = df["Volume"]

    # Create the shifted lag series of prior trading period close values
	for i in range(0, lags):
		tslag["Lag%s" % str(i+1)] = df["Adj Close"].shift(i+1)


    
	dfret = pd.DataFrame(index=tslag.index)
	dfret["Volume"] = tslag["Volume"]
	dfret["Today"] = tslag["Today"].pct_change()*100

	print (dfret)
	for i in range(0, lags):
		dfret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100

	dfret["Direction"] = np.sign(dfret["Today"])


	dfret.drop(dfret.index[:6], inplace=True)
	print (dfret)

	return dfret


if __name__ == "__main__":

	data = create_dataset('AAPL', ('2012-1-1'), ('2017-5-31'), lags=5)

	X = data[["Lag1","Lag2","Lag3","Lag4"]]
	y = data["Direction"]

	start_test = pd.to_datetime('2017-1-1')

	X_train = X[X.index < start_test]
	X_test = X[X.index >= start_test]
	y_train = y[y.index < start_test]
	y_test = y[y.index >= start_test]

	model = LogisticRegression()

	model.fit(X_train, y_train)

	pred = model.predict(X_test)

	print("Accuracy of logistic regression model: %0.3f" % model.score(X_test, y_test))
	print("Confusion matrix: \n%s" % confusion_matrix(pred, y_test))
