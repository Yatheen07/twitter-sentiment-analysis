# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 18:23:46 2018

@author: kmy07
"""

import re
import numpy as np
import string
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns

def remove_pattern(input_txt,pattern):
    
    r = re.findall(pattern,input_txt)

    for i in r:
        input_txt = re.sub(i,'',input_txt)
    
    return input_txt

def hashTag_extract(x):
    hashtags = [] 
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
        
    return hashtags

def PreprocessData(combined_data):
   
    combined_data['tidy_tweet'] = np.vectorize(remove_pattern)(combined_data['tweet'],'@[\w]*')
    combined_data['tidy_tweet'] = combined_data['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
    combined_data['tidy_tweet'] = combined_data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    
    return combined_data

def tokeniseData(combined_data):

    tokenized_tweet = combined_data['tidy_tweet'].apply(lambda x:x.split())
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    return tokenized_tweet

def getWordCloud(words):
    from wordcloud import WordCloud
    wordcloud = WordCloud(width=1000,height=1000,random_state=21,max_font_size=110).generate(words)
    plt.figure(figsize=(12,12))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
def getDataFrame(List):
    dataFrame = pd.DataFrame({'HashTag':list(List.keys()),
                      'Count' : list(List.values())
                    })
    dataFrame.nlargest(columns="Count" , n = 15)
    return dataFrame

def plotDataFrame(dataFrame):
    plt.figure(figsize = (20,10))
    sns.barplot(data = dataFrame, x = "HashTag" , y =  "Count")
    plt.show()
    
def get_Bag_Of_Words(combined_data):
    from sklearn.feature_extraction.text import CountVectorizer
    bag_of_words = CountVectorizer(max_df = 0.90,
                               min_df = 2,
                               max_features = 1500,
                               stop_words = 'english'
                              )
    
    bag_of_words = bag_of_words.fit_transform(combined_data['tidy_tweet'])
    return bag_of_words