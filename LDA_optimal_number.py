#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv('sample_100000_v4.csv')


# In[ ]:


from nltk.corpus import gutenberg
import text_normalizer as tn
import nltk
from operator import itemgetter
from gensim.models import word2vec


# In[ ]:


import pandas as pd
import numpy as np
#df = pd.read_csv('uk_job(other variables cleaned).csv')
df1 = df['job_description']

train_string = np.array(df1)

wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(document) for document in train_string]


# In[ ]:


import gensim

bigram = gensim.models.Phrases(tokenized_corpus, min_count=20, threshold=20) # higher threshold fewer phrases.
bigram_model = gensim.models.phrases.Phraser(bigram)


# In[ ]:


norm_corpus_bigrams = [bigram_model[doc] for doc in tokenized_corpus]

# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
print('Sample word to number mappings:', list(dictionary.items())[:15])
print('Total Vocabulary Size:', len(dictionary))


# In[ ]:


# Filter out words that occur less than 20 documents, or more than 60% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)
print('Total Vocabulary Size:', len(dictionary))


# In[ ]:


bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
print(bow_corpus[1][:50])


# In[ ]:


from tqdm import tqdm

def topic_model_coherence_generator(corpus, texts, dictionary, 
                                    start_topic_count=2, end_topic_count=10, step=1,
                                    cpus=1):
    
    models = []
    coherence_scores = []
    for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
        lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=100000, 
                                               alpha='auto', eta='auto', random_state=42,
                                               iterations=500,num_topics=topic_nums, 
                                               passes=20,eval_every=None)
        cv_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=corpus, 
                                                                     texts=texts, dictionary=dictionary, 
                                                                     coherence='c_v')
        coherence_score = cv_coherence_mode_lda.get_coherence()
        coherence_scores.append(coherence_score)
        models.append(lda_model)
    
    return models, coherence_scores


# In[ ]:


lda_models, coherence_scores = topic_model_coherence_generator(corpus=bow_corpus, texts=norm_corpus_bigrams,
                                                               dictionary=dictionary, start_topic_count=5,
                                                               end_topic_count=30, step=1, cpus=16)


# In[ ]:


coherence_df = pd.DataFrame({'Number of Topics': range(5, 31, 1),
                             'Coherence Score': np.round(coherence_scores, 4)})
#coherence_df.sort_values(by=['Coherence Score'], ascending=False).head(10)


# In[ ]:


coherence_df.to_csv('LDA_coherence_score.csv')

