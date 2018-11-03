from sklearn import naive_bayes
from sklearn.metrics import accuracy_score

import numpy as np


#############TRAINING DATA###################
with open('data/ds1/ds1Train.csv', 'r') as file:
    train_data = [line.split(',') for line in file.read().split('\n')]

#get rid of newline character at end of list
if(train_data[-1] == ['']):
    train_data.pop()

train_features = np.array([d[:-1] for d in train_data], dtype=np.int32)
train_labels = np.array([d[-1] for d in train_data], dtype=np.int32)


#########VALIDATION DATA ########################
with open('data/ds1/ds1Val.csv', 'r') as file:
    validation_data = [line.split(',') for line in file.read().split('\n')]

#get rid of newline character at end of list
if(validation_data[-1] == ['']):
    validation_data.pop()

validation_features = np.array([d[:-1] for d in validation_data], dtype=np.int32)
validation_labels = np.array([d[-1] for d in validation_data], dtype=np.int32)


classifier = naive_bayes.BernoulliNB()
classifier.fit(train_features, train_labels)

validation_predicted = classifier.predict(validation_features)


accuracy = accuracy_score(validation_labels, validation_predicted)
print(accuracy)
