# -*- coding: utf-8 -*-
from solution import SentimentAnalyzer
import cPickle
from nltk.classify.naivebayes import NaiveBayesClassifier
path_to_training_corpus = "../trainset.html"
path_to_model = "solution.pkl"

# train model
sa = SentimentAnalyzer(path_to_training_corpus)

# save trained model to file (send this file with other sources)
f = open(path_to_model, "wb")
cPickle.dump(sa, f)
f.close()
