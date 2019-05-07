#!/usr/bin/env python
# coding: utf-8

# In[8]:


from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd


# In[10]:


numpy.random.seed(7)
train = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/traindata_scaled.csv")
test = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/testdata_scaled.csv")


# In[11]:


train.columns


# In[12]:


test.columns


# In[13]:


train= train.drop(['Unnamed: 0'], axis=1)
test = test.drop(['Unnamed: 0'], axis = 1)


# In[14]:


X = train.drop(['amount_spent_per_room_night_scaled'], axis=1)
Y = train['amount_spent_per_room_night_scaled'];


# In[22]:


model = Sequential()
model.add(Dense(12, input_dim=21,activation='relu'))
model.add(Dense(4, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))


# In[23]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:


model.fit(X, Y, epochs=10, batch_size=10)


# In[25]:


predictions = model.predict(test)


# In[26]:


max= 10.81665
min= 1.600397


# In[27]:


predictions = pd.DataFrame(predictions*(max-min)+min)


# In[28]:


predictions.to_csv("fin6.csv")


# In[ ]:




