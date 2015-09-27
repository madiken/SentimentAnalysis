# -*- coding: utf-8 -*-
from deserializer import deserialize
import cPickle
import time

#from sklearn.metrics import accuracy_score
def my_accuracy_score(answers,pred):
    errors = 0.0 
    for i in range(len(answers)):
      if (answers[i] != pred[i]):
          errors = errors + 1.0
    return 1 - errors/len(answers)
path_to_model = "solution.pkl"
path_to_test_corpus = "../tests/testset.html"

# load trained classifier
f = open(path_to_model)
sa = cPickle.load(f)
f.close()

# prepare testset
test_pairs = deserialize(path_to_test_corpus)
texts = map(lambda x: x[0], test_pairs)
answers = map(lambda x: x[1], test_pairs)

# test!
t1 = time.time()
pred = sa.getClasses(texts)
t2 = time.time()
print(t2-t1)
print("pred " , pred)
print("answers ", answers)
print my_accuracy_score(answers,pred)

