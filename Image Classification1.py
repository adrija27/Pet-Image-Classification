#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


datadir=r"C:\Users\ADRIJA\Desktop\kagglecatsanddogs_3367a\PetImages"
categories=["Dog","Cat"]
for category in categories:
    path=os.path.join(datadir,category) # path to cats or dogs dataset
    class_num=categories.index(category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        break
    break                                # running for loop for only one time just to check the image working


# In[10]:


print(img_array.shape) # 3D for r,g and b


# In[11]:


imsize=100
new_array=cv2.resize(img_array,(imsize,imsize))
plt.imshow(new_array,cmap='gray')


# In[12]:


#create data
training_data=[]
def create_training_data():
    for category in categories:
        path=os.path.join(datadir,category) # path to cats or dogs dataset
        class_num=categories.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # as image color is not a decidng factor
                new_array=cv2.resize(img_array,(imsize,imsize))   # resizing all images evenly
                training_data.append([new_array,class_num])       # appending image data and it's corresponding class
            except Exception as e:                                # exception hadling because some images are broken
                pass
create_training_data()


# In[148]:


print(len(training_data))


# In[14]:


#shuffling the data because the first 12473 datas are of dogs and the rest of cats
import random
random.shuffle(training_data) 


# In[15]:


for sample in training_data[:10]:
    print(sample[1])


# In[149]:


# preparing the data so that it can be feeded into the RandomForestModel
y=np.array(y)
x=[]        
y=[]
for feat,label in training_data:
    x.append(feat)
    y.append(label)
x=np.array(x)                     
X=[]
for i in x:
    i=i.reshape(i.shape[0]*i.shape[1],)
    X.append(i)
X=np.array(X)


# In[150]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[151]:


# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=2,random_state=0)
rf.fit(X_train,Y_train)


# In[152]:


d=rf.predict(X_test) # predicting the test data


# In[153]:


from sklearn.metrics import accuracy_score      # checking for accuracy
print("Accuracy: ",accuracy_score(Y_test,d))

