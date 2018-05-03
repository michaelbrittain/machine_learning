# Data Munging

## Tools: Python, Spark, R, Excel

1. Identify and account for missing values
2. Recognise and fix corrupt data

# Feature Extraction

## Tools: Natural Language Processing, Image and Video Processing

1. Unstructured or semantically complex
2. Different forms - image, video, text

# Dimensionality Reduction

1. Principle component analysis
2. Feature selection techniques

# Feature Engineering

1. Create more relevant features from raw features

# Choose Algorithm

## Classification - output falls into a set list of categories (goal: assign a category) + relates to one instance; gives black blox

### Algorithms: Naive Bayes, Support Vector Machines, Tree based models, Logistic Regression, Decision Trees, Random Forests, K-based Neighbours

1. Spam detection - is this email spam or ham?
2. Sentiment analysis - is this tweet's sentiment positive or negative?
3. Trading strategy - is this trading day going to be an up-day or a down-day?

## Regression - output is a continuous value, dependent on a number of factors (goal: compute a continuous value) + relates to relationship as a whole; gives formula

### Algorithms: Linear Regression, Non-linear Regression

1. Demand forecasting - what will be the *sales* of this product in a given week?
2. Predicting stock returns - what will be the *price* of (or *returns* from) a stock on a given date?
3. How *long* will it take to *commute* from point A to point B?
4. If waiting time increases, *how does this affect* customer satisfaction?
5. Use linear regression to find the beta of a stock

## Clustering - classification/groups is unknown; operates on group of instances, not 1 instance (unlike classification); unsupervised learning (no training phase); choose attributes based on the insights you're looking for

### Algorithms: K-Means, Hierarchichal, Density Based, Distribution Based

1. Segment users into meaningful groups
2. Group users based on similar usage patterns - freq of morning/evening login, time spent per session
3. Group users based on likes/shares
4. What kind of groups can these users be divided into?
5. What kinds of themes are present in this set of articles?

## Recommendations/personalisation/collaborative filtering - use only a user's past behaviour to determine what they might like/need (ignores product attributes); solves discovery & engagement

### Collaborative Filtering Algorithms: Alternating Least Squares, Nearest Neighbour Model
### Other Algorithms: Association Rules, Content Based Filtering

1. What kind of *artists* will this user like?
2. What are the *top 10 book picks* for this user?
3. If a user buys this phone, *what else* will they buy?
4. Inbox organisation - Gmail auto-tagging important emails
5. Feed organisation - Facebook, BBC homepage

# Algorithm Tweaking

1. Hyper Parameter Tuning
2. Cross Validation
3. Ensembling
