#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor, LinearRegression

# find the Beta (volatility/risk) of google stock = (stock/google return - bond/risk free return) / (market/nasdaq return - bond/risk free return)
def readFile(filename, tbond=False):
    data = pd.read_csv(filename, sep=',', usecols=[0, 5], names=['Date', 'Price'], header=0) 
    if tbond:
        data['Returns'] = data['Price']/100
    else:
        #get return by dividing current months price with previous months price, by offsetting array
        returns = np.array(data['Price'][:-1], np.float) / np.array(data['Price'][1:], np.float)-1
        data['Returns'] = np.append(returns, np.nan)
    data.index = data['Date']
    data = data['Returns'][0:-1]  # remove last row as no change data for last item
    return data

googData = readFile('./data/google.csv')
nasdaqData = readFile('./data/nasdaq.csv')
tbondData = readFile('./data/tbond5yr.csv', tbond=True)

reg = SGDRegressor(eta0=0.1, max_iter=100000, fit_intercept=False)
reg.fit((nasdaqData-tbondData).values.reshape(-1,1), (googData-tbondData))

print(reg.coef_)
