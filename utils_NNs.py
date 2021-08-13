#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import nltk
from nltk.stem.porter import *
import pandas as pd
from keras.models import Sequential
from keras import layers, optimizers, regularizers
from keras.layers import Dense, Dropout, Lambda
from keras.layers import Embedding, Input, Flatten
from keras.models import Model
import numpy as np
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import ShuffleSplit
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# In[ ]:


url = "https://tfhub.dev/google/elmo/2"
embed = hub.Module(url)


# In[19]:


def read_csv_file(path):
    data = pd.read_csv(path, index_col=0)
    return data


# In[ ]:


def split_data(X, y_categorical):
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    X_train1 = X_train1.reset_index(drop=True)
    
    return X_train1, X_test1, y_train1, y_test1


# In[20]:


def labels_to_categorical(data):
    X = data.sentence
    y = data.label
    y_cat = to_categorical(y)

    return X, y, y_cat


# In[ ]:


def plot_class_distribution(data):
    
    X,y,y_cat = labels_to_categorical(data)
    X_train1, X_test1, y_train1, y_test1 = split_data(X, y_cat, test_size=0.2, random_state=42)

    y_train2 = pd.DataFrame(y_train1)
    y_train2 = pd.to_numeric(y_train2.idxmax(axis=1))

    unique, counts = np.unique(y_train2, return_counts=True)
    
    colors = ['c', 'm', 'y']
    labelss = ['DN','SN','WN']
    explode = (0.08, 0, 0)
    
    
    plt.pie(counts, colors=colors, labels=labelss,
    explode=explode, autopct='%1.1f%%',
    counterclock=False, shadow=True)
    plt.title('class distribution train set')
    plt.legend(labelss,loc=3)
    plt.savefig('class_distribution_in_training_set.png')
    
    y_test2 = pd.DataFrame(y_test1)
    y_test2 = pd.to_numeric(y_test2.idxmax(axis=1))

    unique_, counts_ = np.unique(y_test2, return_counts=True)
    
    plt.pie(counts_, colors=colors, labels=labelss,
    explode=explode, autopct='%1.1f%%',
    counterclock=False, shadow=True)
    plt.title('class distribution test set')
    plt.legend(labelss,loc=3)
    plt.savefig('class_distribution_in_test_data.png')


# # models

# In[26]:


def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def mlp_elmo_model(learning_rate): 
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
    dense = Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))(embedding)
    dense = Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.005))(dense)
    pred = Dense(3, activation='softmax')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    opt = optimizers.rmsprop(lr= learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


# In[ ]:


def glove_emb_matrix(glove_path, X):

    embeddings_index = dict()
    f = open(path+'/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    t, vocab_size = tokenizer(X)
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in t.word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
                
    return embedding_matrix


def bi_lstm_glove(glove_path, X):
    
    embedding_matrix = glove_emb_matrix(glove_path, X)
    
    input = Input(shape=(max_length,))
    model = Embedding(vocab_size,100,weights=[embedding_matrix],input_length=max_length, trainable=False)(input)
    model =  Bidirectional (LSTM (16,return_sequences=True,dropout=0.3),merge_mode='concat')(model)
    model = TimeDistributed(Dense(16,activation='relu'))(model)
    model = Flatten()(model)
    #model = Dense(50,activation='relu')(model)
    output = Dense(3,activation='softmax')(model)
    model = Model(input,output)
    opt = optimizers.rmsprop(lr=0.0008)
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

    return model


# In[27]:


def evaluate_model(trained_model, test_set, true_values):
    pred = trained_model.predict(test_set)
    predi = pd.DataFrame(pred)
    predi = pd.to_numeric(predi.idxmax(axis=1))
    
    true_val = pd.DataFrame(true_values)
    true_val = pd.to_numeric(true_val.idxmax(axis=1))
    
    acc = accuracy_score(true_val, predi)
    report = classification_report(true_val, predi)
    
    return acc, report


# In[ ]:





# In[2]:


def train_model_n_fold_cv(model, data, n_split, test_size, batch_size): #model()
    all_acc = []
    all_val_acc = []
    all_loss = []
    all_val_loss = []

    fold = 1
    
    cv = ShuffleSplit(n_splits=n_split, random_state=42, test_size=test_size)
    
    X, y, y_cat = labels_to_categorical(data)
    
    X_train1, X_test1, y_train1, y_test1 = split_data(X, y_cat, test_size=0.2, random_state=42)
    
    for train_index, test_index in cv.split(X_train1, y_train1):
        X_train, X_val = X_train1[train_index], X_train1[test_index]
        y_train, y_val = y_train1[train_index], y_train1[test_index]

        model = model()
        history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val))

        acc = history.history['acc']
        loss = history.history['loss']

        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']


        all_acc.append(acc)
        all_loss.append(loss)

        all_val_acc.append(val_acc)
        all_val_loss.append(val_loss)


        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, color='red', label='Training loss')
        plt.plot(epochs, val_loss, color='green', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        plt.plot(epochs, acc, color='red', label='Training acc')
        plt.plot(epochs, val_acc, color='green', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        fold += 1
    
    acc, report = evaluate_model(model, X_test1, y_test1)
    
    save_model(model, "model.h5")
    with open("model_arch.json", 'w') as f:
        f.write(model.to_json())
        
        
    return model, acc, report


# # Grid-Search

# In[4]:


def parameters_grid(learning_rates, batch_sizes):
    param_grid = {'learning_rate':list(learning_rates), 'batch_size':list(batch_sizes)}

    return param_grid

def grid_search(model,learning_rates,batch_sizes, data):
    model = KerasClassifier(build_fn=model, verbose=1, epochs=20)
    param_grid = parameters_grid(learning_rates,batch_sizes)
    grid = GridSearchCV(estimator=model1, param_grid=param_grid, cv=cv, return_train_score=True)
    
    X_train, X_test, y_train, y_test = split_data(X, y_cat, test_size=0.2, random_state=42)
    
    X_train1 = np.array(X_train)
    y_train1 = to_categorical(y_train)
    
    grid_result = grid.fit(X_train1, y_train1)
        
    return grid_result

def find_best_param_in_grid(model,learning_rates,batch_sizes, data):
    grid_result = grid_search(model,learning_rates,batch_sizes, data)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    return grid_result.best_score_, grid_result.best_params_


# In[ ]:





# In[ ]:




