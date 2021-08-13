#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.stem.porter import *
import string
import re, string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, zero_one_loss
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
import warnings
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit


# In[ ]:


def read_csv_file(path):
    data = pd.read_csv(path, index_col=0)
    data.label.replace([0, 1, 2], ['DN', 'SN', 'WN'], inplace=True)
    
    return data


# In[2]:


def strip_links(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def clean_stemm(text):
    stemmer = PorterStemmer()
    text = " ".join(re.split("\s([@#][\w_-]+)", text.lower())).strip()
    text = ' '.join(stemmer.stem(token) for token in nltk.word_tokenize(text))
    return text

def clean_lemma(text):
    lemmatizer = WordNetLemmatizer()
    text = " ".join(re.split("[^a-zA-Z]*", text.lower())).strip()
    text = " ".join(re.split("\s([@#][\w_-]+)", text)).strip()
    text = ' '.join(lemmatizer.lemmatize(token) for token in nltk.word_tokenize(text))
    return text


# In[3]:


def preprocess(data):
    le = LabelEncoder()
    
    strip_text = [strip_links(i) for i in data.sentence]
    normalize  = [clean_lemma(i) for i in strip_text]
    
    data['sentence'] = normalize
    data['label'] = le.fit_transform(data['label'])
    
    return data


# In[ ]:





# In[67]:


text = data.sentence
len(text)


# In[ ]:


def to_csv(data, path):
    data.to_csv(path)


# In[ ]:


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


# # BASELINE

# In[2]:


def tfidf_unigram(X, y):
    vectorizer_u = TfidfVectorizer(tokenizer=tokenize,ngram_range=(0,1))
    tfidf_u = vectorizer_u.fit_transform(X).toarray()
    X_train1, X_test1, y_train1, y_test1 = split_data(tfidf_u, y)
    
    return X_train1, X_test1, y_train1, y_test1


def tfidf_bigram_topn(X, y, topn=None):
    vectorizer_b = TfidfVectorizer(tokenizer=tokenize,ngram_range=(1, 2))
    tfidf_b = vectorizer_b.fit_transform(X).toarray()
    
    tfidf_b_reduced = SelectKBest(k=topn).fit_transform(tfidf_b, y)
    X_train1, X_test1, y_train1, y_test1 = split_data(tfidf_b_reduced, y)
    
    return X_train1, X_test1, y_train1, y_test1


def tfidf_trigram_topn(X, y, topn=None):
    vectorizer_t = TfidfVectorizer(tokenizer=tokenize,ngram_range=(1, 3))
    tfidf_t = vectorizer_t.fit_transform(X).toarray()
    
    tfidf_t_reduced = SelectKBest(k=topn).fit_transform(tfidf_t, y)
    X_train1, X_test1, y_train1, y_test1 = split_data(tfidf_t_reduced, y)
    
    return X_train1, X_test1, y_train1, y_test1
    


# # TFIDF-Unigram

# In[8]:


def modelling(X,y,model, tfidf_ngram):

    mean_train_acc = []
    mean_val_acc = []
    
    X_train1, X_test1, y_train1, y_test1 = tfidf_ngram(X,y)
    
    y_train1 = np.array(y_train1)
    y_test1 = np.array(y_test1)
    
    
    cv = ShuffleSplit(n_splits=5, random_state=42, test_size=0.2)
    
    fold = 1
    for train_index, test_index in cv.split(X_train1, y_train1):
        X_train, X_val = X_train1[train_index], X_train1[test_index]
        y_train, y_val = y_train1[train_index], y_train1[test_index]
        #print(X_train.shape, y_train.shape,  X_test.shape, y_test.shape)

        #clf = LogisticRegression(random_state=0)

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)


        training_acc = clf.fit(X_train, y_train).score(X_train, y_train)
        val_acc = accuracy_score(pred, y_val)

        training_loss = 1.0 - training_acc
        val_loss = zero_one_loss(y_val, pred)

        fold += 1
        
        
    pred = clf.predict(X_test1)
    acc = accuracy_score(y_test1, pred)
    report = classification_report(y_test1, pred)

    return acc, report


# In[ ]:


if __name__ == "__main__":
    
    data = read_csv_file(path)
    preprocess_data = preprocess(data)
    X = preprocess_data.sentence
    y = preprocess_data.label
    names = ["Logistic Regression", "SVC (kernal=linear)"]
    classifiers = [LogisticRegression(random_state=0),
                   SVC(kernel="linear",random_state=0)]
    X,y = 
    for name, clf in zip(names, classifiers):    
        train_model = train_model_nfold_cv(X,y,clf, tfidf_ngram)
        print('\x1b[1;31m'+name+'\x1b[0m')
        print(acc)
        print(report)

