#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


train = pd.read_csv(r"C:\Users\Amisha\Desktop\train.csv")


# In[5]:


test = pd.read_csv(r"C:\Users\Amisha\Desktop\test.csv")


# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


train.columns


# In[9]:


test.columns


# In[11]:


## We now know that 'subscribed' is the target variable


# In[15]:


train.shape, test.shape


# In[16]:


train.dtypes


# ### Univariate Analysis

# In[21]:


train['subscribed'].value_counts()


# In[22]:


train['subscribed'].value_counts().plot.bar()


# In[24]:


sns.distplot(train["age"])


# In[25]:


train['job'].value_counts().plot.bar()


# In[26]:


train['marital'].value_counts().plot.bar()


# In[27]:


train['education'].value_counts().plot.bar()


# In[28]:


train['default'].value_counts().plot.bar()


# ### Bivariate Analysis

# In[29]:


pd.crosstab(train['age'],train['subscribed'])


# In[33]:


pd.crosstab(train['age'],train['subscribed'])


# In[34]:


pd.crosstab(train['education'],train['subscribed'])


# In[35]:


pd.crosstab(train['marital'],train['subscribed'])


# In[36]:


pd.crosstab(train['default'],train['subscribed'])


# In[37]:


train.corr() ## only for numerical variables


# In[39]:


train.isnull().sum()


# In[52]:


## since there are no null values we can procede


# In[53]:


## sk learn works only for numerical values


# In[54]:


train = pd.get_dummies(train)


# # MODEL BUILDING

# ## LOGISTIC REGRESSION

# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


logreg = LogisticRegression()


# In[ ]:


y = train['subscribed']
train = train.drop('subscribed',1)


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


x_train, x_val, y_train, y_val = train_test_split(train, y, test_size = 0.2, random_state=12)


# In[60]:


logreg.fit(x_train,y_train)


# In[61]:


prediction=logreg.predict(x_val)


# In[64]:


from sklearn.metrics import accuracy_score


# In[65]:


accuracy_score(y_val,prediction)


# In[66]:


## let us see if we can have a better algorithm for the same


# In[67]:


from sklearn.tree import DecisionTreeClassifier


# In[69]:


df=DecisionTreeClassifier(max_depth=10)


# In[72]:


df.fit(x_train,y_train)


# In[73]:


df.fit(x_val,y_val)


# In[75]:


test = pd.get_dummies(test)


# In[78]:


predict = df.predict(x_val)


# In[79]:


accuracy_score(y_val,predict)


# In[80]:


test_predict=df.predict(test)


# In[81]:


test_prediction=logreg.predict(test)


# In[87]:


submission12 = pd.DataFrame()


# In[88]:


submission12['ID'] = test['ID']
submission12['subscribed'] = test_prediction


# In[89]:


submission12['subscribed'].replace(0,'no',inplace=True)
submission12['subscribed'].replace(1,'yes',inplace=True)


# In[93]:


submission12.to_csv('submission12.csv', header=True, index=False)


# In[92]:


submission22 = pd.DataFrame()

submission22['ID'] = test['ID']
submission22['subscribed'] = test_predict

submission22['subscribed'].replace(0,'no',inplace=True)
submission22['subscribed'].replace(1,'yes',inplace=True)

submission22.to_csv('submission22.csv', header=True, index=False)


# In[ ]:




