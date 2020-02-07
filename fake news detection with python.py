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




lemztr = WordNetLemmatizer()
# df['text2_stemmed2_lem'] = df['text2_stemmed2'].apply(lambda x: [lemztr.lemmatize(y) for y in x])
# df['title2_stemmed2_lem'] = df['title2_stemmed2'].apply(lambda x: [lemztr.lemmatize(y) for y in x])


# In[18]:


df2=df[["label","text2_stemmed2","title2_stemmed2"]]


# In[24]:


df2.head()


# In[50]:


# ngrams
# TextBlob(df2['text2_stemmed2'][5]).ngrams(3)


# In[55]:


wc = WordCloud(background_color="white", max_words=2000, width=800, height=400)
# generate word cloud
wc.generate(' '.join(df2['text2_stemmed2']))

# show
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# further cleaning is required for unwanted terms


# In[ ]:


# sentiment analysis
# coming up...


# In[ ]:





# In[58]:


# define labels
labels=df2.label
labels.head()


# In[59]:


x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[60]:


# tfidf vectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[61]:


pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[62]:


confusion_matrix(y_test,y_pred, labels=['fake','real'])


# In[63]:


# naive bayes classifier


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# topic modeling


# In[76]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(df2['text2_stemmed2'].values.astype('U'))


# In[77]:


from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(doc_term_matrix)


# In[78]:


import random

for i in range(10):
    random_id = random.randint(0,len(count_vect.get_feature_names()))
    print(count_vect.get_feature_names()[random_id])


# In[79]:


first_topic = LDA.components_[0]
top_topic_words = first_topic.argsort()[-10:]
for i in top_topic_words:
    print(count_vect.get_feature_names()[i])


# In[80]:


# 10 words with highest probabilities for five topics
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[84]:


topic_values = LDA.transform(doc_term_matrix)
topic_values.shape


# In[83]:


# add a topic column
df2['topic'] = topic_values.argmax(axis=1)
df2.head()


# In[ ]:
