import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf

yf.pdr_override()

stock = ['AAPL']
start = pd.to_datetime('2013-12-12')
end = pd.to_datetime('2018-03-29')
data = pdr.get_data_yahoo(stock, start=start, end=end)['Adj Close']
daily_returns = (data/data.shift(1))-1
daily_returns.hist(bins=100)
plt.show()
