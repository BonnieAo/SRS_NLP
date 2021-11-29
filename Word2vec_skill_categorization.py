#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from gensim.models import word2vec

df = pd.read_csv('/Users/aoziqiao/Desktop/new_uk_job_other_ariables_cleaned.csv')
model_w2v= word2vec.Word2Vec.load("/Users/aoziqiao/Desktop/word2vec_200_ukjob.model")
disco = pd.read_csv('disco_new_normalized.csv')

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm

def average_word_embedding(s):
    s = s.split()
    #vector = np.zeros(300,)
    vector = model_w2v.wv[s[0]]
    vector1 = vector.copy()
    length = len(s)
    for i in range (1, len(s)):
        try:
            vector1 +=model_w2v.wv[s[i]]
        except:
            i+=1
            length -=1
    vector1 = vector1/length
    return vector1

def ave_Word2vec_cos_similarity(s1, s2):
    vector1 = average_word_embedding(s1)
    vector2 = average_word_embedding(s2)
    # similarity
    return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))


# In[ ]:


def similarity_classification (s):
    category = "a"
    disco['similarity'] = disco["Key Words"].apply(lambda x: ave_Word2vec_cos_similarity(x,s))
    argmax = disco[disco['similarity'] == disco['similarity'].max()].index[0]
    category = disco['domain specific skills and competences'][argmax]
    print(1)
    return category


# In[ ]:


df['skill_category'] = 'a'
df['skill_category'] = df["job_description"].apply(lambda x: similarity_classification(x))


# In[ ]:


df.to_csv('/Users/aoziqiao/Desktop/uk_job_skill_category.csv')

