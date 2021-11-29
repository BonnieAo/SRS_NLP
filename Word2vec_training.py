#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
disco = pd.read_csv('/Users/aoziqiao/Desktop/disco_new_normalized.csv')
df = pd.read_csv('/Users/aoziqiao/Desktop/new_uk_job_other_ariables_cleaned.csv')
df1 = disco['Key Words']
df2 = df['job_description']
df3 = df1.append(df2)
train_string = np.array(df3)


# In[ ]:


from gensim.models import word2vec
import nltk

# tokenize sentences in corpus
wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in train_string]

# Set values for various parameters
feature_size = 200    # Word vector dimensionality  
window_context = 30          # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
sample = 1e-3   # Downsample setting for frequent words

w2v_model= word2vec.Word2Vec(tokenized_corpus, vector_size = feature_size, window=window_context, min_count=min_word_count, sample=sample,epochs = 100)

w2v_model.save("/Users/aoziqiao/Desktop/word2vec_200_ukjob.model")

