#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd


# In[5]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


# In[3]:


train = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/newtraindata.csv")
test = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/newtestdata.csv")


# In[6]:


train.columns


# In[7]:


train= train.drop(['Unnamed: 0'], axis=1)
test = test.drop(['Unnamed: 0'], axis = 1)


# In[ ]:





# In[8]:


X, y = train.iloc[:,:-1],train.iloc[:,-1]


# In[9]:


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)


# In[17]:





# In[ ]:


xg_reg.fit(X,y)


# In[18]:





# In[11]:


preds = xg_reg.predict(test)


# In[ ]:





# In[12]:


preds


# In[ ]:


preds1


# In[13]:


predictions = pd.DataFrame(preds)


# In[ ]:





# In[14]:


predictions.to_csv("xgb.csv")


# In[ ]:




