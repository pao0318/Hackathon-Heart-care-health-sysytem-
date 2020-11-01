#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import seaborn as sns


# In[2]:


df=pd.read_csv('heart.csv')


# In[4]:


df=df.drop(['slope','oldpeak'],axis=1)


# In[5]:


# sns.scatterplot(data=df,x='chol',y='target')


# In[6]:


df=df[df['chol']<400]


# # Training

# In[10]:


from sklearn.model_selection import train_test_split


# In[8]:


X = df.drop("target",axis=1)
y = df["target"]


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[12]:


from sklearn.preprocessing import RobustScaler


# In[13]:


scaler=RobustScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# # Training using various models

# In[14]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(C=0.5)
log=model.fit(X_train,y_train)


# In[15]:


pred=log.predict_proba(X_test)


# In[16]:


pred


# In[17]:


#for i in range(len(pred)):
#   if (pred[i][0]>pred[i][1]):
#        pred[i][1]=0
#   else:
#        pred[i][0]=0
        


# In[18]:


pred[0][1]


# In[19]:


import joblib
joblib.dump(scaler,'scaler.pkl')


# In[20]:


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[21]:


log_scaler=joblib.load('scaler.pkl')


# In[22]:


df[df['target']==0]


# In[23]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# In[24]:


pred1=log.predict(X_test)


# In[25]:


from sklearn.metrics import classification_report,confusion_matrix


# In[26]:


print(classification_report(y_test,pred1))


# In[ ]:





# In[ ]:




