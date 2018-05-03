#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


with open('./data/imdb_labelled.txt', 'r') as text_file:
    lines = text_file.read().split('\n')

with open('./data/yelp_labelled.txt', 'r') as text_file:
    lines += text_file.read().split('\n')

with open('./data/amazon_cells_labelled.txt', 'r') as text_file:
    lines += text_file.read().split('\n')

lines = [line.split("\t") for line in lines if len(line.split("\t"))==2 and line.split("\t")[1]!='']

# split the data into features (comments) and labels (positive/negative boolean flag)
train_documents = [line[0] for line in lines]
train_labels = [int(line[1]) for line in lines]

# convert each comment in list into a word frequency tuple list
count_vectorizer = CountVectorizer(binary='true')
train_documents = count_vectorizer.fit_transform(train_documents)
# print(train_documents[1])

# create model using training data and algorithm
classifier = BernoulliNB().fit(train_documents, train_labels)

# todo - create subset of data for testing and see accuracy stats of model

# test model's predictive ability
predict = classifier.predict(count_vectorizer.transform(["this is the best movie"]))
print(predict)
predict = classifier.predict(count_vectorizer.transform(["this is the worst movie"]))
print(predict)
