#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd


# In[43]:


numpy.random.seed(7)
train = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/traindata_scaled.csv")
test = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/testdata_scaled.csv")


# In[44]:


train.columns


# In[45]:


train= train.drop(['Unnamed: 0'], axis=1)
test = test.drop(['Unnamed: 0'], axis = 1)


# In[46]:


X = train.drop(['amount_spent_per_room_night_scaled'], axis=1)
Y = train['amount_spent_per_room_night_scaled'];


# In[47]:


import statsmodels.api as sm


# In[48]:


X = sm.add_constant(X)


# In[49]:


test = sm.add_constant(test)


# In[50]:


model = sm.OLS(Y,X).fit()


# In[51]:


predictions = model.predict(test)


# In[52]:


predictions


# In[53]:


predictions = pd.DataFrame(predictions)


# In[54]:


predictions.to_csv("outSM.csv")

