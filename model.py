# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 19:12:55 2018

@author: kmy07
"""
import numpy as np
import dataPreProcessing as data

bag_of_words = data.bagOfWords
test_data = data.test_data
train_data = data.train_data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bag_of_words[:31962,:]
test_bow = bag_of_words[39162:,:]

Xtrain,Xtest,Ytrain,Ytest = train_test_split( train_bow,
                                              train_data['label'],
                                              random_state = 36,
                                              test_size = 0.28
                                            )

regressor = LogisticRegression()
regressor.fit(Xtrain,Ytrain)

prediction = regressor.predict_proba(Xtest)

prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)


accuracy = f1_score(Ytest,prediction_int)

print("The Accuracy of the model is : "+str(accuracy*100)+"%")