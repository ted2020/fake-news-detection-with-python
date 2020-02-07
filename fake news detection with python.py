#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import re


# In[2]:


df=pd.read_csv('news.csv')
del df["Unnamed: 0"]
df.sample(10)


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[6]:


df["title"]=df["title"].str.lower()
df["text"]=df["text"].str.lower()
df["label"]=df["label"].str.lower()


# In[7]:


# remove punctuation
df['text'] = df['text'].apply(lambda x: re.sub('[^\w\s]', ' ', x))
df['title'] = df['title'].apply(lambda x: re.sub('[^\w\s]', ' ', x))


# In[8]:


# remove one and two character words
df['text'] = df['text'].apply(lambda x: re.sub(r'\b\w{1,3}\b', '', x))
df['title'] = df['title'].apply(lambda x: re.sub(r'\b\w{1,3}\b', '', x))


# In[9]:


# remove numerical values
df['text'] = df['text'].apply(lambda x: re.sub(r'[0-9]+', '', x))
df['title'] = df['title'].apply(lambda x: re.sub(r'[0-9]+', '', x))


# In[10]:


# \s+ means all empty space (\n, \r, \t)
df['text'] = df['text'].apply(lambda x: re.sub('\s+', ' ', x))
df['title'] = df['title'].apply(lambda x: re.sub('\s+', ' ', x))


# In[13]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')


# In[14]:


# remove stop words
df['text'] = df['text'].apply(lambda text: " ".join(word for word in text.split() if word not in stop))
df['title'] = df['title'].apply(lambda text: " ".join(word for word in text.split() if word not in stop))


# In[16]:


# tokenization df["text"]
from nltk.tokenize import word_tokenize
df["text2"]=df.apply(lambda row: nltk.word_tokenize(row["text"]), axis=1)


# In[17]:


# tokenization df["title"]
# from nltk.tokenize import word_tokenize
df["title2"]=df.apply(lambda row: nltk.word_tokenize(row["title"]), axis=1)


# In[18]:


df['text2_len'] = df.apply(lambda row: len(row['text2']), axis=1)
df['title2_len'] = df.apply(lambda row: len(row['title2']), axis=1)


# In[20]:


# stemming df["text"]
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
df['text2_stemmed'] = df['text2'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[21]:


# stemming df["title"]
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
df['title2_stemmed'] = df['title2'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[23]:


import string
def remove_punctuation(s):
    s = ' '.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

df['title2_stemmed2']=df['title2_stemmed'].apply(remove_punctuation)
df['text2_stemmed2']=df['text2_stemmed'].apply(remove_punctuation)
