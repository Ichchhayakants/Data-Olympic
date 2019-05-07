#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Pandas is used for data manipulation
import pandas as pd
# Read in data and display first 5 rows
train = pd.read_csv('C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/LTFS/newtraindata.csv')
train.head(5)


# In[17]:


print('The shape of our features is:', train.shape)


# In[20]:


test = pd.read_csv('C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/LTFS/newtestdata.csv')


# In[21]:


test.shape


# In[6]:


# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(train)
# Display the first 5 rows of the last 12 columns
train.iloc[:,5:].head(5)


# In[22]:


feature_list = list(train.columns)
print(feature_list)


# In[9]:


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(train['loan_default'])
# Remove the labels from the features
# axis 1 refers to the columns
features= train.drop('loan_default', axis = 1)
# Saving feature names for later use
feature_list = list(train.columns)
# Convert to numpy array
features = np.array(train)


# In[10]:


print(train)


# In[23]:


train = train.drop(['Unnamed: 0'], axis=1)
test = test.drop(['Unnamed: 0'], axis=1)


# In[27]:


train.columns


# In[28]:


test.columns


# In[29]:


x = train.drop(['loan_default'], axis=1)
y = train['loan_default'];


# In[30]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(x,y);


# In[ ]:


import pickle
filename = 'randomforest.sav'
pickle.dump(rf, open(filename, 'wb'))


# In[1]:


# Use the forest's predict method on the test data
predictions = rf.predict(test)
# Calculate the absolute errors
#errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#Mean Absolute Error: 3.83 degrees.


# In[43]:


print(predictions)


# In[53]:


print(type(predictions))


# In[54]:


pred = []
for i in predictions:
    if i < 0.5:
        pred.append(0)
    else:
        pred.append(1)


# In[55]:


print(pred)


# In[57]:


fp = open("out.csv","w")
fp.write(repr(pred))


# In[ ]:





# In[58]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# In[ ]:




