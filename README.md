# Supervised-Learning-Techniques-for-Sentiment-Analytics
Perform sentiment analysis over IMDB movie reviews and Twitter data. Our goal will be to classify tweets or movie reviews as either positive or negative.

We have a labeled training data to build the model and labeled testing data to evaluate the model. For classification, we will experiment with logistic regression as well as a Naive Bayes classifier from python’s well regarded machine learning package scikitlearn.

A major part of this project is the task of generating feature vectors for use in these classifiers. We will explore two methods: 
(1) A more traditional NLP technique where the features are simply “important” words and the feature vectors are simple binary vectors and (2) the Doc2Vec technique where document vectors are learned via artificial neural networks

# Datasets
The IMDB reviews and tweets can be found in the data folder. These have already been divided into train and test sets.
● The IMDB dataset that contains 50,000 reviews split evenly into 25k train and 25k test sets. Overall, there are 25k pos and 25k neg reviews. In the labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets.
● The Twitter Dataset contains 900,000 classified tweets split into 750k train and 150k test sets. The overall distribution of labels is balanced (450k pos and 450k neg).

# Functions in sentiment.py:

● feature_vecs_NLP: A word should be counted at most once per tweet/review even if the word has occurred multiple times in that tweet/review.
● feature_vecs_DOC: Make a list of LabeledSentence objects from the word lists. These objects consist of a list of words and a list containing a single string label. We will want to use a different label for the train/test and pos/neg sets. For example, we used TRAIN_POS_i, T RAIN_NEG_i, T EST_POS_i, and T EST_NEG_i, where i is the line number.
● evaluate_model: Here we calculate the true positives, false positives, true negatives, false negatives, and accuracy.

