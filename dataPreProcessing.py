# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 18:23:18 2018

@author: kmy07
"""

'''Step 1: Import Neccessary Packages'''

import numpy as np
import pandas as pd
import userDefined_functions as func
import seaborn as sns
import nltk

train_data = pd.read_csv('./dataset/train_E6oV3lV.csv')
test_data = pd.read_csv('./dataset/test_tweets_anuFYb8.csv')

combined_data = train_data.append(test_data,ignore_index=True)

combined_data = func.PreprocessData(combined_data)

tokenised_tweet = func.tokeniseData(combined_data)

normal_tweets = combined_data['tidy_tweet'][combined_data['label'] == 0]
racist_tweets = combined_data['tidy_tweet'][combined_data['label'] == 1]

normal_words = ' '.join([words for words in normal_tweets])
racist_words = ' '.join([words for words in racist_tweets])

#func.getWordCloud(normal_words)
#func.getWordCloud(racist_words)

hashtags_normal = func.hashTag_extract(normal_tweets)
hashtags_racist = func.hashTag_extract(racist_tweets)

hashtags_normal = sum(hashtags_normal,[])
hashtags_racist = sum(hashtags_racist,[])

freq_count_normal = nltk.FreqDist(hashtags_normal)
freq_count_racist = nltk.FreqDist(hashtags_racist)

dataFrame_normal = func.getDataFrame(freq_count_normal)
dataFrame_racist = func.getDataFrame(freq_count_racist)

#func.plotDataFrame(dataFrame_normal)
#func.plotDataFrame(dataFrame_racist)

bagOfWords = func.get_Bag_Of_Words(combined_data)
  
