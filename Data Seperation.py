#!/usr/bin/env python
# coding: utf-8

import pandas as pd


# In[42]:

df = pd.read_csv('uk_job(other variables cleaned).csv')

# In[44]:

def list_of_groups(init_list, childern_list_len):
    list_of_groups = zip(*(iter(init_list),) *childern_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list

# In[46]:

l = list(range(len(df)))
a = list_of_groups(l,286628)

def seperation(df,i):
    df1 = df.iloc[a[i][0]:a[i][-1]+1,:]
    dfname = "/Users/aoziqiao/Desktop/"+"df"+str(i)+".csv"
    df1.to_csv(dfname)
    return "1 done!"

for i in range (0:10):
    seperation(df, i)
