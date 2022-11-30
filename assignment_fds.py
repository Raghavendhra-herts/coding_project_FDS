#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


read_data = np.loadtxt("inputdata7.csv", delimiter="," , dtype=str)

# read the data into Python numpy array(s)
read_data = np.loadtxt("inputdata7.csv", delimiter="," , dtype=str, skiprows=1)
print("\n \n", read_data)


# In[5]:


# plot the data as a two-dimensional scatter plot,
x , y = np.transpose(read_data)
print(x)


# In[6]:


print(y)


# In[12]:


plt.scatter(x,y)


# In[7]:


import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.xlim(0,15)


# In[8]:


from sklearn.linear_model import LinearRegression
print(x)


# In[26]:


model = LinearRegression()
x = np.array(x.reshape(-1,1), dtype=float)
print(x)
y = np.array(y, dtype=float)
model.fit(x, y)


# In[29]:


score = model.score(x, y)
score
intercept = model.intercept_
coef = model.coef_
print(score, "\t", intercept, "\t", coef)


# In[30]:


y_pred = model.predict(x)
y_pred


# In[36]:


plt.scatter(x, y, color = 'g')
plt.plot(x, y_pred, color = 'r')
plt.xlabel("Rainfall per year")
plt.ylabel("Field productivity")


# In[ ]:




