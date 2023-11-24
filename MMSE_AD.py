#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc


# In[3]:


data= pd.read_csv("Desktop/OASIS_longitudinal.csv")


# In[59]:


data


# In[60]:


data.shape


# In[5]:


data= data[data["Visit"]==1]


# In[6]:


data.head()


# In[7]:


data= data.reset_index(drop= True)


# In[8]:


data.head()


# In[9]:


data.shape


# In[10]:


data["Group"].value_counts()


# In[11]:


data['M/F'].value_counts()


# In[12]:


data['M/F']= data['M/F'].replace(['F', 'M'], [0,1])
data['Group']= data['Group'].replace(['Converted'], ['Demented'])
data['Group']= data['Group'].replace(['Demented', 'Nondemented'], [1,0])
data = data.drop(['MRI ID', 'Visit', 'Hand'], axis=1)


# In[13]:


data.head()


# In[ ]:





# In[14]:


def make_bar(feature):
    
    Demented= data[data['Group']==1][feature].value_counts()
    Nondemented= data[data['Group']==0][feature].value_counts()
    df_bar= pd.DataFrame([Demented, Nondemented])
    df_bar.index= ['Demented', 'Nondemented']
    df_bar.plot(kind='bar', stacked=True, figsize=(8,5))
    


# In[15]:


make_bar('M/F')


# In[ ]:





# In[16]:


facet= sns.FacetGrid(data, hue="Group", aspect=3)
facet.map(sns.kdeplot, 'MMSE', shade= True)
facet.set(xlim=(0, data["MMSE"].max()))
facet.add_legend()
plt.xlim(15.30,35) #Grande diff


# In[17]:


facet= sns.FacetGrid(data, hue="Group", aspect=3)
facet.map(sns.kdeplot, 'ASF', shade= True)
facet.set(xlim=(0, data["ASF"].max()))
facet.add_legend()
plt.xlim(0.5,2)


# In[18]:


facet= sns.FacetGrid(data, hue="Group", aspect=3)
facet.map(sns.kdeplot, 'eTIV', shade= True)
facet.set(xlim=(0, data["eTIV"].max()))
facet.add_legend()
plt.xlim(900,2200)


# In[19]:


facet= sns.FacetGrid(data, hue="Group", aspect=3)
facet.map(sns.kdeplot, 'nWBV', shade= True)
facet.set(xlim=(0, data["nWBV"].max()))
facet.add_legend()
plt.xlim(0.5,1)


# In[20]:


facet= sns.FacetGrid(data, hue="Group", aspect=3)
facet.map(sns.kdeplot, 'Age', shade= True)
facet.set(xlim=(0, data["Age"].max()))
facet.add_legend()
plt.xlim(50,110)


# In[21]:


facet= sns.FacetGrid(data, hue="Group", aspect=3)
facet.map(sns.kdeplot, 'EDUC', shade= True)
facet.set(xlim=(0, data["EDUC"].max()))
facet.add_legend()
plt.xlim(1,30)


# In[22]:


pd.isnull(data).sum()


# In[23]:


data_dropna = data.dropna(axis=0, how='any') #on enleve les valeurs manquantes


# In[24]:


data_SES= data.groupby(['EDUC'])['SES'].median()


# In[25]:


data_SES #le niveau d'éducation n'a pas un réel impact sur le niveau social (SES)


# In[26]:


data['SES'].fillna(data.groupby("EDUC")['SES'].transform("median"), inplace= True)


# In[27]:


y= data['Group'].values #la méthode values permet de transformer le dataframe en array
                        #pour appliquer des modèles


# In[28]:


y


# In[29]:


X = data[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)


# In[31]:


X_train.head()


# In[32]:


# Feature scaling
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[33]:


X_train_scaled.shape


# ## Régression logistique régularisée

# In[39]:


best_score= 0
k_folds=5

for c in [0.001, 0.1, 1, 10, 100]:
    
    logRegModel= LogisticRegression(C=c)
    scores= cross_val_score(logRegModel, X_train, y_train, cv=k_folds, scoring= 'accuracy')
    score= np.mean(scores)
    
    if score > best_score:
        
        best_score= score
        best_parameter= c
        
Selected_Log_reg= LogisticRegression(C= best_parameter)
Selected_Log_reg.fit(X_train_scaled, y_train)
test_score= Selected_Log_reg.score(X_test_scaled, y_test)
Predicted_Output_test= Selected_Log_reg.predict(X_test_scaled)
test_recall= recall_score(y_test, Predicted_Output_test, pos_label=1)
Predicted_prob_test= Selected_Log_reg.predict_proba(X_test_scaled)
fpr, tpr, thresholds= roc_curve(y_test, Predicted_prob_test[:,1])
test_auc= auc(fpr, tpr)


print("L'accuracy du train set est de", Selected_Log_reg.score(X_train_scaled, y_train))
print("L'accuracy du test set est de", test_score)
print("Le meilleur paramètre de régularisation est", best_parameter)
print("La sensibilité du test (taux de vrais positifs parmi tous les positifs est de", test_recall)
print("L'AUC est de", test_auc)
plt.plot(fpr, tpr)


# ## SVM (Support Vector Machine)

# In[45]:


best_score= 0
k_folds=5

for c_parameter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    for gamma_parameter in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for kernel_parameter in ['rbf', 'linear', 'poly', 'sigmoid']:
            svmModel= SVC(kernel=kernel_parameter, C= c_parameter, gamma= gamma_parameter)
            
            scores= cross_val_score(svmModel, X_train_scaled, y_train, cv= k_folds, scoring= 'accuracy')
            score= np.mean(scores)
            
            if score > best_score:
                
                best_score= score
                best_parameter_c= c_parameter
                best_parameter_gamma= gamma_parameter
                best_parameter_kernel= kernel_parameter
                
                
SelectedSVMmodel= SVC(C= best_parameter_c, gamma= best_parameter_gamma, kernel= best_parameter_kernel, probability= True).fit(X_train_scaled, y_train)

test_score= SelectedSVMmodel.score(X_test_scaled, y_test)
PredictedOutput= SelectedSVMmodel.predict(X_test_scaled)
test_recall= recall_score(y_test, PredictedOutput, pos_label=1)
PredictedOutput_proba= SelectedSVMmodel.predict_proba(X_test_scaled)
fpr, tpr, thresholds= roc_curve(y_test, PredictedOutput_proba[:,1])
test_auc= auc(fpr, tpr)

print("Meilleur paramètre c: ", best_parameter_c)
print( "Meilleur paramètre gamma", best_parameter_gamma)
print("Meilleur paramètre kernel: ", best_parameter_kernel)
print("Test accuracy: ", test_score)
print("Valeur de la sensitivité:", test_recall)
print("Valeur de l'AUC", test_auc)


# ## Decision tree

# In[52]:


best_score= 0
k_folds=5

for md in range(1,9):
    
    treeModel= DecisionTreeClassifier(random_state=0, max_depth= md, criterion= 'gini')
    
    scores= cross_val_score(treeModel, X_train_scaled, y_train, cv= k_folds, scoring= "accuracy")
    score= np.mean(scores)
    
    if score > best_score:
        
        best_score= score
        best_parameter= md
        
Selected_DT_model= DecisionTreeClassifier(max_depth= best_parameter).fit(X_train_scaled, y_train)

test_score= Selected_DT_model.score(X_test_scaled, y_test)
PredictedOutput= Selected_DT_model.predict(X_test_scaled)
test_recall= recall_score(y_test, PredictedOutput, pos_label=1)
PredictedOutput_proba= Selected_DT_model.predict_proba(X_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test, PredictedOutput_proba[:,1], pos_label=1)
test_auc= auc(fpr, tpr)

print("Meilleur paramètre pour max_depth :", best_parameter)
print("Test accuracy: ", test_score)
print("Sensitivité :", test_recall)
print("AUC :", test_auc)


# ## Random Forest

# In[56]:


best_score= 0
k_folds=5

for M in range(2,15,2):
    for d in range(1,9):
        for m in range(1,9):
            
            forestModel= RandomForestClassifier(n_estimators=M, max_features= d, n_jobs= 4, max_depth=m, random_state=0)
            
            scores= cross_val_score(forestModel, X_train_scaled, y_train, cv= k_folds, scoring= 'accuracy')
            
            score= np.mean(scores)
            
            if score > best_score:
                
                best_score= score
                best_M= M
                best_d= d
                best_m=m
                
SelectedRFModel = RandomForestClassifier(n_estimators=best_M, max_features=best_d,
                                          max_depth= best_m, random_state=0).fit(X_train_scaled, y_train)   


PredictedOutput = SelectedRFModel.predict(X_test_scaled)
test_score = SelectedRFModel.score(X_test_scaled, y_test)
test_recall = recall_score(y_test, PredictedOutput, pos_label=1)
PredictedOutput_proba= SelectedRFModel.predict_proba(X_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test, PredictedOutput_proba[:,1], pos_label=1)
test_auc= auc(fpr, tpr)

print("Best parameters of M, d, m are: ", best_M, best_d, best_m)
print("Test accuracy :", test_score)
print("Sensitivité :", test_recall)
print("AUC:", test_auc)


# ## Adaboost

# In[58]:


best_score= 0
k_folds= 5


for M in range(2,15,2):
    for lr in [0.0001, 0.001, 0.01, 0.1, 1]:
        
        boostModel= AdaBoostClassifier(n_estimators= M, learning_rate= lr, random_state=0)
        
        scores= cross_val_score(boostModel, X_train_scaled, y_train, cv= k_folds, scoring= 'accuracy')
        
        score= np.mean(scores)
        
        if score > best_score:
            
            best_score= score
            best_M= M
            best_lr= lr
            
            
SelectedBoostModel= AdaBoostClassifier(n_estimators= best_M, learning_rate= lr, random_state=0).fit(X_train_scaled, y_train)

PredictedOutput= SelectedBoostModel.predict(X_test_scaled)
test_score= SelectedBoostModel.score(X_test_scaled, y_test)
test_recall= recall_score(y_test, PredictedOutput, pos_label=1)
PredictedOutput_proba= SelectedBoostModel.predict_proba(X_test_scaled)
fpr, tpr, thresholds= roc_curve(y_test, PredictedOutput_proba[:,1], pos_label=1)
test_auc= auc(fpr, tpr)


print("best parameter M: ", best_M)
print("best parameter lr:", best_lr)
print("test accuracy:", test_score)
print("sensitivité: ", test_recall)
print("AUC:", test_auc)






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




