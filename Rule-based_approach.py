#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import nltk
import spacy
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


def negatively_polarized_lexicon(negative_polarity_words_txt_file_path):
    
    neg = open(negative_polarity_words_txt_file_path)
    f2 = neg.read()
    f2 = f2.replace('\n', ' ')
    negative_polarity_words = nltk.word_tokenize(f2)

    return negative_polarity_words


# In[1]:


def read_csv(path_test_set):
    csv = pd.read_csv(path_test_set)
    return csv


# In[ ]:


def evaluation(prediction, test_csv):
    
    true_values = []
    for i in test_csv.label:
        if i == 0:
            true.append(i)
        else:
            true.append(1)
     
    acc = accuracy_score(true_values, prediction)
    report = classification_report(true_values, prediction)
    
    return acc, report


# In[ ]:


# Detect double negation in a statement.


# In[2]:


def double_negation_detector(csv, negative_polarity_words):
    
    nlp = spacy.load('en_core_web_sm')
    
    all_relation = []
    all_dn = []

    neg_words = negative_polarity_words
    WHOLEWORD = ["wasn","can","none", "never", "no", "nobody","cannot", "n't", "not", "without", "nowhere", "nothing", "neither", "minus", "nor", "don", "weren", "isn", "doesn", "couldn", "shouldn", "hasn", "won", "wouldn", "can","can not", "aren","nor", "haven","hadn", "didn","isn't'","is not"]

    for n, i in enumerate(csv.sentence):
        relation = []
        dn = []
        doc = nlp(i)
        for token in doc:
        #################_____RULE NO. 1   
            try:
                if token.dep_ == 'ROOT' or token.pos_ == 'VERB' :             # A token is root Verb that is not negative
                    neg = [w for w in token.head.children if w.dep_ == 'neg']    #and it has a child node with dependecy relation tag "neg"/ contains negative adverb.
                    if neg:
                        try:                                       # same token has another child node which contains a negative polarity word.
                            acc = [w for w in token.head.children if w.lemma_ in neg_words and w.pos_ != "VERB"]
                            if acc:
                                if (neg,acc) not in relation:
                                    relation.append((neg,acc,rule))
                                    dn.append((n,0))
                                    break
                        except:
                            pass

                        try:                                      # A root token contains a negative adverb i.e Disagree,
                            if token.lemma_ in neg_words:          #and token has child node which also contains negative adverb.
                                rel = [token]
                                if rel:
                                    if (neg,rel) not in relation:
                                        relation.append((neg,rel))
                                        dn.append((n,0))
                                        break
                        except:
                            pass

            except:
                continue


        #################_____RULE NO. 2 - non neg ROOT has child and grand child
            try:
                if token.dep_ == 'ROOT':             # if a token which is root But not a negative adverb, 
                    neg = [w for w in token.head.children if w.dep_ == 'neg']    #and it has a child node with dependecy relation tag "neg"/ contains negative adverb.
                    if neg:
                        for w in token.children:        # - same token has child node and that node also has a child node which contains neg_polarity word
                                                        # - or we can say, NON NEG, ROOT token has a grand child node which contains negative pol word.
                            rel1 = [w for w in w.children if w.lemma_ == "no"] 
                            if rel1:
                                if (neg1,rel1) not in relation:
                                    relation.append((neg1,rel1))
                                    dn.append((n,0))
                                    break
            except:
                pass


        #################_____RULE NO. 3
            try:
                if token.pos_ != "VERB":                      # A token that is not root or Verb and not negative adverb,
                    for w in token.head.children:             # but it has a child node which contains negative adverb / "neg" relation,
                        if w.lemma_ in WHOLEWORD:
                            neg1 = [w]
                            
                            if neg1:                        # and the same token has another child node that contains negative word.
                                rel1 = [w for w in token.head.children if w.lemma_ in neg_words and w.lemma_ != "no"]

                                if rel1 and rel1 != neg1:
                                    if (neg1,rel1) not in relation:
                                        relation.append((neg1,rel1)) 
                                        dn.append((n,0))
                                        break

            except:
                continue


        #################_____RULE NO. 4        
            try:
                if token.lemma_ in neg_words:               #If a token which is not root and but is a negative polarity word,
                    rel = [token]
                    if rel:                              #and token has a child node that contains negative adverb / dependecy relation tag "neg"
                        neg = [w for w in token.children if w.dep_ == "neg"]
                        if neg:
                            if (neg,rel) not in relation:
                                relation.append((neg,rel)) 
                                dn.append((n,0))
                                break
            except:
                continue


        #################_____RULE NO. 5           
            try: 
                if token.pos_ != "VERB":    #if a token which is not root and also not a negative adverb, 
                    for w in token.head.children: 
                        if w.dep_ == "neg":    # - but it has a child node which with dependecy relation tag "neg"/negative adverb and 
                            neg1 = [w]
                            if neg1: 
                                try:                  # - same token has another child node which contains a negative polarity word
                                    rel1 = [w for w in token.head.children if w.lemma_ in neg_words]
                                    if rel1 and rel1 != neg1:
                                        if ((neg1,rel1)) not in relation:
                                            relation.append((neg1,rel1)) 
                                            dn.append((n,0))
                                            break
                                except:
                                    pass
                                for w in token.children:        # - same token has child node and that node also has a child node whhich contains neg_polarity word
                                    if w.children:              # - or we can say, NON NEG, NON ROOT token has a grand child node which contains negative pol word.
                                        if w.lemma_ in neg_words:
                                            rel1 = [w]
                                            if rel1:
                                                if (neg1,rel1) not in relation:
                                                    relation.append((neg1,rel1))
                                                    dn.append((n,0))
                                                    break


            except:
                continue

        if len(dn) > 0:
            all_relation.append(relation)
            prediction.append(0)
        else:
            prediction.append(1)
            
    return prediction


# In[ ]:


if __name__ == "__main__":
    
    negative_polarity_words = negatively_polarized_lexicon('negative_polarity_words.txt')
    csv = read_csv('test.csv')
    
    prediction = double_negation_detector(csv, negative_polarity_words)
    
    evaluate = evaluation(prediction, csv)


# In[ ]:


"""
To visualze tree on server

#text = (u"I cannot disagree with your point of view.")
#doc = nlp(text)
#for token in doc:
   # print(token.lemma_, token.pos_, token.tag_, token.dep_,token.is_stop)
#from spacy import displacy
#displacy.render(doc, style='dep',jupyter=True)
#spacy.displacy.serve(doc, style='dep')
"""

