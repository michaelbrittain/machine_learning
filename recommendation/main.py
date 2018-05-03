#!/usr/bin/env python
# -*- coding: utf-8 -*-
import implicit
import pandas as pd
from scipy.sparse import coo_matrix
import heapq


# read first 3 columns into a dataframe
dataFile = './data/u.data'
data = pd.read_csv(dataFile, sep='\t', header=None, usecols=[0,1,2], names=['userId','itemId','rating']) 
# print(data.head())

# convert dataframe to matrix of users v items, with ratings as values
data['userId'] = data['userId'].astype('category')
data['itemId'] = data['itemId'].astype('category')
rating_matrix = coo_matrix((
        data['rating'].astype(float),
        (
            data['itemId'].cat.codes.copy(),
            data['userId'].cat.codes.copy(),            
        )
    ))

# split matrix into users v factors and factors v items matrices, using als algorithm
model = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.1)
model.fit(rating_matrix)
user_factors, item_factors = model.item_factors, model.user_factors

# to get the predicted ratings for all movies by this user:
#   take one row from the user_factors matrix - which represents one user
#   then take dot product of that row with all the columns in the item factors
user196 = item_factors.dot(user_factors[196])

# then sort these ratings in descending order and pick the top3 rated movies recommended for this user
recommendations = heapq.nlargest(3, range(len(user196)), user196.take)
print(recommendations)
