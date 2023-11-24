#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
import random


# In[2]:


Mild_demented=[]
for i in tqdm(range(1,897)):
    filename= f"Desktop/Dataset/Mild_Demented/mild_{i}.jpg"
    img= Image.open(filename)
    #nouvelle_largeur= img.width // 4
    #nouvelle_hauteur= img.height // 4
    #img_redim= img.resize((nouvelle_largeur, nouvelle_hauteur))
    tableau_pixels= np.asarray(img)
    Mild_demented.append(tableau_pixels.tolist())


# In[3]:


Mild_demented= np.array(Mild_demented)


# In[4]:


Mild_demented.shape #image de taille 128 fois 128 pixels


# In[5]:


Non_demented=[]
for i in tqdm(range(1,3201)):
    filename= f"Desktop/Dataset/Non_Demented/non_{i}.jpg"
    img= Image.open(filename)
    #nouvelle_largeur= img.width // 4
    #nouvelle_hauteur= img.height // 4
    #img_redim= img.resize((nouvelle_largeur, nouvelle_hauteur))
    tableau_pixels= np.asarray(img)
    Non_demented.append(tableau_pixels.tolist())


# In[6]:


Non_demented= np.array(Non_demented)


# In[7]:


Very_Mild_demented=[]
for i in tqdm(range(1,2241)):
    filename= f"Desktop/Dataset/Very_Mild_Demented/verymild_{i}.jpg"
    img= Image.open(filename)
    #nouvelle_largeur= img.width // 4
    #nouvelle_hauteur= img.height // 4
    #img_redim= img.resize((nouvelle_largeur, nouvelle_hauteur))
    tableau_pixels= np.asarray(img)
    Very_Mild_demented.append(tableau_pixels.tolist())


# In[8]:


Very_Mild_demented= np.array(Very_Mild_demented)


# In[9]:


X= np.concatenate((Mild_demented, Non_demented, Very_Mild_demented), axis=0)


# In[10]:


X= X.reshape(-1,128,128,1)


# In[11]:


y=[]
for i in range(896):
    y.append(0)
for i in range(3200):
    y.append(1)
for i in range(2240):
    y.append(2)


# In[12]:


y=np.array(y)


# In[13]:


permut= np.random.permutation(len(X))


# In[14]:


X_bis= X[permut]


# In[15]:


y_bis= y[permut]


# In[16]:


X_train, X_test, y_train, y_test= train_test_split(X_bis, y_bis, test_size=0.2, random_state=42)


# In[17]:


X_train= X_train/255.0
X_test= X_test/255.0


# In[18]:


model= keras.models.Sequential()

model.add(keras.layers.Input((128,128,1)))

model.add(keras.layers.Conv2D(8, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(3, activation= 'softmax'))


# In[19]:


model.compile(loss='sparse_categorical_crossentropy', optimizer= 'adam',metrics=['accuracy'])


# In[26]:


batch_size= 50
epochs= 20

history= model.fit(X_train, y_train, batch_size= batch_size, epochs=epochs, verbose=1, validation_data= (X_test, y_test))


# In[29]:


model.summary()


# In[28]:


score= model.evaluate(X_test, y_test)

print(score[0]) #test loss
print(score[1]) 


# In[ ]:





# In[ ]:





# In[20]:


X_train.shape


# In[82]:


filename= "Desktop/Dataset/Mild_Demented/mild_1.jpg"
img= Image.open(filename)


# In[83]:


img


# In[84]:


tableau_pixels= np.asarray(img)
tableau_pixels= tableau_pixels.reshape((1,128,128,1))
tableau_pixels.shape


# In[85]:


y_pred= model.predict(tableau_pixels)
etiq= np.argmax(y_pred, axis=1)

if (etiq[0]==0):
    print("Prédiction: Mild-demented")
    
if (etiq[0]==1):
    print("Prédiction: Non-demented")
    
if(etiq[0]==2):
    print("Prédiction: Very Mild demented")


# In[86]:


filename_bis= "Desktop/Dataset/Non_Demented/non_1.jpg"
img_bis= Image.open(filename_bis)


# In[87]:


img_bis


# In[88]:


tableau_pixels= np.asarray(img_bis)
tableau_pixels= tableau_pixels.reshape((1,128,128,1))
tableau_pixels.shape


# In[89]:


y_pred= model.predict(tableau_pixels)
etiq= np.argmax(y_pred, axis=1)

if (etiq[0]==0):
    print("Prédiction: Mild-demented")
    
if (etiq[0]==1):
    print("Prédiction: Non-demented")
    
if(etiq[0]==2):
    print("Prédiction: Very Mild demented")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




