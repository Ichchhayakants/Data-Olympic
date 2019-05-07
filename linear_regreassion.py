#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd


# In[2]:


numpy.random.seed(7)
train = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/newtraindata.csv")
test = pd.read_csv("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra/newtestdata.csv")


# In[3]:


train= train.drop(['Unnamed: 0'], axis=1)
test = test.drop(['Unnamed: 0'], axis = 1)


# In[4]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[5]:


x = train.drop(['amount_spent_per_room_night_scaled'], axis=1)
y = train['amount_spent_per_room_night_scaled'];


# In[6]:


model = LinearRegression().fit(x, y)


# In[7]:


r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)


# In[8]:


print('intercept:', model.intercept_)


# In[9]:


print('slope:', model.coef_)


# In[11]:


y_pred = model.predict(test)


# In[12]:


print('predicted response:', y_pred, sep='\n')


# In[13]:


predictions = pd.DataFrame(y_pred)


# In[14]:


predictions.to_csv("LinReg.csv")


# In[ ]:


# Polynomial regression


# In[15]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


#Transform input data


# In[16]:


transformer = PolynomialFeatures(degree=2, include_bias=False)


# In[17]:


transformer.fit(x)


# In[18]:


x_ = transformer.transform(x)


# In[19]:


x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)


# In[20]:


print(x_)


# In[ ]:


# Create a odel and fit it


# In[21]:


model = LinearRegression().fit(x_, y)


# In[22]:


r_sq = model.score(x_, y)


# In[23]:


print('coefficient of determination:', r_sq)


# In[24]:


print('intercept:', model.intercept_)


# In[25]:


print('coefficients:', model.coef_)


# In[26]:


x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)


# In[27]:


print(x_)


# In[28]:


print(x_)


# In[29]:


r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)


# In[30]:


print('intercept:', model.intercept_)


# In[31]:


print('coefficients:', model.coef_)


# In[35]:


test_= PolynomialFeatures(degree=2, include_bias=True).fit_transform(test)


# In[ ]:





# In[34]:


y_pred = model.predict(test_)


# In[ ]:


print('predicted response:', y_pred, sep='\n')


# In[ ]:


#Advanced Linear Regression With statsmodels


# In[39]:


import numpy as np
import statsmodels.api as sm


# In[40]:


x = sm.add_constant(x)


# In[41]:


model = sm.OLS(y, x)


# In[42]:


results = model.fit()


# In[43]:


print(results.summary())


# In[44]:


print('coefficient of determination:', results.rsquared)


# In[45]:


print('adjusted coefficient of determination:', results.rsquared_adj)
print('regression coefficients:', results.params)


# In[46]:


print('predicted response:', results.fittedvalues, sep='\n')
print('predicted response:', results.predict(x), sep='\n')


# In[47]:


y_new = results.predict(test)

