#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


num_clusters = 5

with open('./data/imdb_labelled.txt', 'r') as text_file:
    lines = text_file.read().split('\n')

lines = [line.split("\t") for line in lines if len(line.split("\t"))==2 and line.split("\t")[1]!='']

# extract just features (comments)
train_documents = [line[0] for line in lines]

# convert each comment into its tfidf representation
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
train_documents = tfidf_vectorizer.fit_transform(train_documents)
# print(train_documents[1])

# create model using training data and algorithm
km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(train_documents)

# see the clusters generated
for cluster in range(num_clusters):
    count = 0
    print('cluster', cluster)
    for i in range(len(lines)):
        if count > 3:
            break
        if km.labels_[i] == cluster:
            print(lines[i])
            count += 1

