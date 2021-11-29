#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.corpus import gutenberg
import text_normalizer as tn
import nltk
from operator import itemgetter
from gensim.models import word2vec
import pandas as pd
df = pd.read_csv('/Users/aoziqiao/Desktop/sample_ukjob_500000.csv').drop_duplicates()


# In[2]:


import pandas as pd
import numpy as np
#df = pd.read_csv('uk_job(other variables cleaned).csv')
df1 = df['job_description']

train_string = np.array(df1)

wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in train_string]


# In[3]:


import gensim

bigram = gensim.models.Phrases(tokenized_corpus, min_count=20, threshold=20) # higher threshold fewer phrases.
bigram_model = gensim.models.phrases.Phraser(bigram)


# In[4]:


norm_corpus_bigrams = [bigram_model[doc] for doc in tokenized_corpus]

# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
print('Sample word to number mappings:', list(dictionary.items())[:15])
print('Total Vocabulary Size:', len(dictionary))


# In[5]:


# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)
print('Total Vocabulary Size:', len(dictionary))


# In[6]:


# Transforming corpus into bag of words vectors
bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
print(bow_corpus[1][:50])


# In[ ]:


#%%time
TOTAL_TOPICS = 24
lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=100000, 
                                   alpha='auto', eta='auto', random_state=42,
                                   iterations=500, num_topics=TOTAL_TOPICS, 
                                   passes=20, eval_every=None)


for topic_id, topic in lda_model.print_topics(num_topics=24, num_words=30):
    print('Topic #'+str(topic_id+1)+':')
    print(topic)
    print()
    
    
topics_coherences = lda_model.top_topics(bow_corpus, topn=24)
avg_coherence_score = np.mean([item[1] for item in topics_coherences])
print('Avg. Coherence Score:', avg_coherence_score)

df = pd.DataFrame([i for i in range(1,31)])

topics_with_wts = [item[0] for item in topics_coherences]

for idx, topic in enumerate(topics_with_wts):
    df['topic'+str(idx+1)] = [term for wt, term in topic]
    df['wt'+str(idx+1)] = [wt for wt, term in topic]

df = df.drop(columns = [0])
df


# In[ ]:


df.to_csv('topic_weight_LDA_wholedata.csv')

