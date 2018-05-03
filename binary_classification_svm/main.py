#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

# load data into panda dataframe
dataFile = './data/ad.data'
data = pd.read_csv(dataFile, sep=',', header=None, low_memory=False)
# print(data.head(20))

# change missing values to NaN
def toNum(cell):
    try:
        return np.float(cell)
    except:
        return np.nan

# convert ad text labels to boolean
def toLabel(str):
    if str == 'ad.':
        return 1
    else:
        return 0

# apply missing column check to a column / panda series
def seriestoNum(series):
    return series.apply(toNum)

# apply check to every column in the dataframe, except the last (the labels), then drop those rows
train_data = data.iloc[0:, 0:-1].apply(seriestoNum)  # row0 to end; col0 to end-1 
train_data = train_data.dropna()
# print(train_data.head(20))

train_labels = data.iloc[train_data.index, -1].apply(toLabel)  # ensure aligned: only row indexes within training data; column last
# print(train_labels)

# train svc model using subset of data
clf = LinearSVC()
clf.fit(train_data[100:2300], train_labels[100:2300])

# todo - create subset of data for testing and see accuracy stats of model

# predict new image
predict = clf.predict(train_data.iloc[12].reshape(1, -1))
print(predict)

predict = clf.predict(train_data.iloc[-1].reshape(1, -1))
print(predict)
