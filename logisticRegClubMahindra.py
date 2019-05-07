#!/usr/bin/env python
# coding: utf-8

# In[9]:


from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd


# In[10]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


# In[15]:


train = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/traindata_scaled.csv")
test = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/testdata_scaled.csv")


# In[16]:


train= train.drop(['Unnamed: 0'], axis=1)
test = test.drop(['Unnamed: 0'], axis = 1)


# In[14]:


train.head


# In[17]:


X, y = train.iloc[:,:-1],train.iloc[:,-1]


# In[34]:


xg_reg1 = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 10,gamma=0.4,alpha = 10, n_estimators = 10,subsample=0.8)


# In[38]:


xg_reg1 = xgb.XGBRegressor(learning_rate=0.1, n_estimators=200, max_depth=4, min_child_weight=7, 
                      gamma=0.4,nthread=4, subsample=0.8, colsample_bytree=0.8, objective= 'reg:logistic',scale_pos_weight=3,seed=29)


# In[39]:


xg_reg1.fit(X,y)


# In[40]:


preds1 = xg_reg1.predict(test)


# In[41]:


preds1


# In[ ]:





# In[42]:


max= 10.81665
min= 1.600397


# In[43]:


preds1 = preds1*(max-min)+min


# In[44]:


predictions = pd.DataFrame(preds1)


# In[45]:


predictions.to_csv("xgb2.csv")

