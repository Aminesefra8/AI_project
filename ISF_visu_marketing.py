#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[2]:


data= pd.read_csv("Desktop/ifood_df.csv")


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.isna().sum()


# In[6]:


data.info()


# In[7]:


data.nunique()


# In[8]:


data.drop(columns=['Z_CostContact','Z_Revenue'],inplace=True)


# In[9]:


plt.figure(figsize=(15,8))
sns.boxplot(data= data, y="MntTotal")
plt.ylabel("Montant total des clients")
plt.title("Boxplot pour le montant total")
plt.show() 


#On remarque que moins de 25% des clients ont dépensé moins de 200
#50% des clients ont dépensé moins de 400
#75% des clients ont dépensé moins de 1000
#Puis il y a une très petite minorité de client qui ont dépensé plus de 2000 


# ## Outliers

# In[10]:


Q1= data["MntTotal"].quantile(0.25)
Q3= data["MntTotal"].quantile(0.75)
lower_bound= Q1-1.5*(Q3-Q1)
upper_bound= Q3+1.5*(Q3-Q1)
outliers= data[(data["MntTotal"] < lower_bound) | (data["MntTotal"] > upper_bound)]
outliers


#ces outliers vont être enlevés de la base de données car ils ne sont
#pas représentatifs de la tendance des dépenses des clients
#Sur à peu près 2000 clients dans la base de données seulement 3 outliers apparaissent
#Le montant des dépenses de ces clients sont bien au dessus ou bien en dessous
#des dépenses des autres clients qui sont majoritaires, et en marketing on ne 
#peut pas se focaliser sur chaque client un à un, il faut plutôt segmenter
#en groupe de population


# In[11]:


data= data[(data["MntTotal"] > lower_bound) | (data["MntTotal"] < upper_bound)]
data.describe()

#Base de données sans les outliers


# In[12]:


plt.figure(figsize=(15,8))
sns.boxplot(data= data, y= "Income")
plt.ylabel("revenu des clients")
plt.title("Boxplot des revenus des clients")
plt.show()

#On remarque que moins de 25% des clients ont des revenus inférieures à 35000
#50% des clients ont des revenus inférieures à 50000
#75% des clients ont des revenus inférieures à 65000


# In[13]:


Q1= data["Income"].quantile(0.25)
Q3= data["Income"].quantile(0.75)
lower_bound= Q1-1.5*(Q3-Q1)
upper_bound= Q3+1.5*(Q3-Q1)
outliers= data[(data["Income"] < lower_bound) | (data["Income"] > upper_bound)]
outliers

#Pas d'outliers


# In[14]:


plt.figure(figsize=(15,8))
sns.histplot(data=data, x='Income', bins=30, kde=True)
plt.title("Histogramme des revenus des clients")
plt.xlabel("Revenus")
plt.ylabel("Fréquence")
plt.show()

#Les revenus des clients semblent suivre une loi normale
#On remarque que la grande majorité des clients ont des revenus compris
#entre 35000 et 75000


# In[15]:


plt.figure(figsize=(15,8))
sns.histplot(data=data, x='Age', bins=30, kde=True)
plt.title("Histogramme de l'âge des clients")
plt.xlabel("Age")
plt.ylabel("Fréquence")
plt.show()

#On remarque que les personnes âgées entre 40 et 50 ans sont les clients les 
#plus majoritaires.
#Même les clients qui ont la trentaine ou la soixantaine sont assez nombreux
#mais ne représentent cependant pas la majorité des clients


# In[16]:


print("Moyenne de l'age des clients:", data['Age'].mean())
print("Skewness de l'age des clients:", data['Age'].skew())
print("Kurtosis de l'age des clients:", data['Age'].kurt())
print(data['Age'].quantile(0.5))

#On a une skewness strictement positive mais proche de 0, donc la distribution de
#l'âge est légérement asymétrique vers la droite, cela peut s'expliquer
#certainement par le fait qu'il y a certains clients assez agés

#Et un kurtosis négatif légérement proche de 0 implique que la distribution 
#de l'âge se rapproche d'une loi normale
# avec les extrémités de la distribution de l'âge qui
#converge vite vers 0 plus vite que la loi normale
#ceci s'explique par le fait qu'il y a très peu de clients
#jeunes (la vingtaine) et très peu de clients très agés (80 ans et plus)


# ## Correlation des variables

# In[17]:


cols_demographics = ['Income','Age']
cols_children = ['Kidhome', 'Teenhome']
cols_marital = ['marital_Divorced', 'marital_Married','marital_Single', 
                'marital_Together', 'marital_Widow']
cols_mnt = ['MntTotal', 'MntRegularProds','MntWines', 'MntFruits', 
            'MntMeatProducts', 'MntFishProducts', 
            'MntSweetProducts', 'MntGoldProds']
cols_communication = ['Complain', 'Response', 'Customer_Days']
cols_campaigns = ['AcceptedCmpOverall', 'AcceptedCmp1', 'AcceptedCmp2', 
                  'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
cols_source_of_purchase = ['NumDealsPurchases', 'NumWebPurchases',
                           'NumCatalogPurchases', 'NumStorePurchases', 
                           'NumWebVisitsMonth']
cols_education = ['education_2n Cycle', 'education_Basic', 
                  'education_Graduation', 'education_Master', 'education_PhD']


# In[18]:


corr_matrix = data[['MntTotal']+cols_demographics+cols_children].corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matrice de corrélation")
plt.show()

#On remarque que le montant total du client est fortement corrélé positivement 
#avec le revenu du client donc on peut en déduire que lorsque les revenus du client
#augmentent et plus le montant total du client augmente

#On remarque aussi que le montant total du client est moyennement corrélé 
#négativement avec le nombre d'enfants à la maison
#Cela impliquerait que plus il y a d'enfants dans la famille du client et plus le 
#montant total des dépenses du client diminue.
#Cela parait assez étrange, mais on peut trouver plusieurs explications 
#sociologiques.
# 1ère explication: Dans de nombreuses familles, les coûts liés aux enfants, 
#tels que l'alimentation, l'habillement et les loisirs, peuvent être 
#partagés entre les enfants. Par conséquent, à mesure que le nombre 
#d'enfants augmente, les dépenses individuelles par enfant peuvent diminuer, 
#ce qui entraîne une baisse des dépenses totales.
#2ème explication: Les familles avec plus d'enfants peuvent adopter des 
#styles de vie différents, tels que des achats en vrac ou des choix 
#budgétaires plus stricts, ce qui peut réduire les dépenses globales. 
#Les parents peuvent être plus enclins à rechercher des offres et à 
#limiter les dépenses superflues.
#3ème explication: Les familles avec plus d'enfants peuvent avoir 
#des priorités financières différentes. Par exemple, elles peuvent 
#consacrer une plus grande partie de leur budget aux besoins de base 
#tels que l'éducation et la santé, ce qui réduit les dépenses dans 
#d'autres domaines.





# In[19]:


#On remarque une corrélation bisérale très faible (proche de 0) entre les montants
#totaux dépensés par les clients et leur situation maritale 
#(divorce, marriage, célibat, couple non marié, veuve...)
#Par exemple, les montants totaux dépensés en moyenne par les clients veufs ne 
#différent pas beaucoup des montants dépensés moyens des clients "non-veufs".
#Le raisonnement est le même pour chaque statut marital.

for col in cols_marital:
    correlation, p_value = pointbiserialr(data[col], data['MntTotal'])
    print(f'{correlation: .4f}: Corrélation bisérale pour {col} avec p-value de {p_value : .4f}')


# In[20]:


#On remarque une corrélation bisérale très faible (proche de 0) entre les montants
#totaux dépensés par les clients et leur niveau d'éducation
#(Licence, Lycée, Master, doctorat)
#Par exemple les montants dépensés en moyenne par les doctorants est très proche
#du montant moyen dépensé par les étudiants n'ayant pas effectués un doctorat.
#Le raisonnement est le même pour niveau d'éducation.


for col in cols_education:
    correlation, p_value = pointbiserialr(data[col], data['MntTotal'])
    print(f'{correlation: .4f}: Corrélation bisérale pour {col} avec p-value de {p_value : .4f}')


# ## Feature engineering

# In[21]:


def get_marital_status(row):
    
    if row["marital_Divorced"]==1:
        return "Divorced"
    elif row["marital_Married"]==1:
        return "Married"
    elif row["marital_Single"]==1:
        return "Single"
    elif row["marital_Together"]==1:
        return "Together"
    elif row["marital_Widow"]==1:
        return "Widow"
    else:
        return "Unknown"


# In[22]:


data["Marital"]= data.apply(get_marital_status, axis=1)


# In[23]:


#Il y a 477 clients célibataires
#854 mariés
#230 divorcés
#568 en couple
#76 veufs(ves)


print("Il ya",data[data["Marital"]=='Widow'].shape[0], "veufs ou veuves parmi les clients")
print("Il ya",data[data["Marital"]=='Together'].shape[0], "en couple parmi les clients")
print("Il ya",data[data["Marital"]=='Divorced'].shape[0], "divorcés parmi les clients")
print("Il ya",data[data["Marital"]=='Married'].shape[0], "mariés parmi les clients")
print("Il ya",data[data["Marital"]=='Single'].shape[0], "célibataires parmi les clients")




# In[132]:


#On remarque que les clients veufs sont les moins nombreux dans la base de donnée
#et pourtant la somme des montants totaux des clients veufs est bien supérieure
#à la somme des montants totaux des clients divorcés, de même pour les clients
#marriés, en couples et célibataires.




plt.figure(figsize=(15,8))
sns.barplot(x="Marital", y="MntTotal", data= data, palette='viridis')
plt.title("Montant total par statut marital")
plt.xlabel("Status marital")
plt.ylabel("Montant total")
plt.show()


# In[25]:


#On va créer une nouvelle feature qui nous dit si le client est dans une 
#relation ou non, cele nous servira plus tard dans notre analyse.

def in_relationship(row):
    
    if row["marital_Married"]==1:
        return 1
    elif row["marital_Together"]==1:
        return 1
    else:
        return 0


# In[26]:


data["In_relationship"]= data.apply(in_relationship, axis=1)


# In[27]:


data["In_relationship"]


# ## Modèle de ML (K-means clustering)

# In[28]:


#On va tout d'abord standardiser les données
#Ici on ne va s'intéresser qu'aux colonnes suivantes: salaires, Montants totaux
#et le situation maritale des clients

scaler= StandardScaler()
cols_for_clustering= ['Income', 'MntTotal', 'In_relationship']
data_scaled= data.copy()
data_scaled[cols_for_clustering]= scaler.fit_transform(data[cols_for_clustering])


# In[29]:


data_scaled[cols_for_clustering].describe()


# In[33]:


pca = decomposition.PCA(n_components = 2)
pca_res = pca.fit_transform(data_scaled[cols_for_clustering])
data_scaled['pc1'] = pca_res[:,0]
data_scaled['pc2'] = pca_res[:,1]


# In[34]:


data_scaled


# In[58]:


X = data_scaled[cols_for_clustering]
inertia_list = []
for K in range(2,10):
    inertia = KMeans(n_clusters=K, random_state=7).fit(X).inertia_
    inertia_list.append(inertia)


# In[59]:


#On calcule l'inertie intra-cluster en fonction du nombre de clusters choisi
#pour le modèle. On remarque que l'inertie baisse moins significativement
#à partir d'un nombre de cluster égale à 4, en général quand la courbe forme une 
#sorte de coude comme on peut le voir dans la zone où le nombre de cluster égale
#à 4, cela veut dire en pratique que le nombre de cluster est optimal.
#On remarque alors que 4 et 5 semble être les nombres de clusters optimaux.
#On peut également choisir un nombre de cluster où l'inertie intra est plus
#petite comme par exemple un nombre de cluster égal à 7, mais il faut aussi
#prendre un nombre de cluster pas trop grand. En effet, à la fin on a envie 
#de segmenter plusieurs types de clients afin d'adapter les publicités, et les offres
#en fonction de ces différents groupes. Si le nombre de cluster est trop grand
#on va devoir adapter des offres et des publicités pour de nombreux groupes de clients
#ce qui peut-être coûteux et fastidieux, en terme de publicité par exemple.
#On se limitera donc à un nombre de cluster égale à 4 ou 5.



plt.figure(figsize=[15,8])
plt.plot(range(2,10), inertia_list)
plt.title("Inertia vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()


# In[64]:


#On voit bien que pour un nombre de clusters égal à 4, on obtient le 
#silhouette score le plus grand de même pour la elbow method.
#On choisira alors un nombre de cluster égal à 4.




silhouette_score_list=[]
for K in range(2,10):
    model= KMeans(n_clusters= K, random_state=7)
    cluster= model.fit_predict(X)
    silhouette= silhouette_score(X, cluster)
    silhouette_score_list.append(silhouette)
    
plt.figure(figsize=(15,8))
plt.plot(range(2,10), silhouette_score_list)
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()


# In[71]:


#Modèle choisi

model_final= KMeans(n_clusters=4, random_state=7)
cluster= model_final.fit_predict(X)
data_scaled["Cluster"]= cluster


# In[72]:


data_scaled


# In[77]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='pc1', y='pc2', data=data_scaled, hue='Cluster', palette='viridis')
plt.title('Clustered Data Visualization')
plt.xlabel('Principal Component 1 (pc1)')
plt.ylabel('Principal Component 2 (pc2)')
plt.legend(title='Clusters')


# In[79]:


data['Cluster'] = data_scaled.Cluster
data.groupby('Cluster')[cols_for_clustering].mean()


# In[83]:


mnt_data = data.groupby('Cluster')[cols_mnt].mean().reset_index()
mnt_data.head()


# In[ ]:





# In[ ]:





# In[84]:


melted_data = pd.melt(mnt_data, id_vars="Cluster", var_name="Product", value_name="Consumption")


# In[86]:





# In[89]:


plt.figure(figsize=(15,8))
sns.barplot(x="Cluster", y="Consumption", hue="Product", data=melted_data, ci=None, palette="viridis")
plt.title("Product Consumption by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Product Consumption")
plt.xticks(rotation=0)  
plt.legend(title="Product", loc="upper right")
plt.show()


# In[103]:


cluster_sizes = data.groupby('Cluster')[['MntTotal']].count().reset_index()
plt.figure(figsize=(15,8))
sns.barplot(x='Cluster', y='MntTotal', data=cluster_sizes, palette = 'viridis')
plt.title('Cluster sizes')
plt.xlabel('Cluster')
plt.ylabel('Count')


# In[113]:


cluster_sizes['Poucentage']= round((cluster_sizes['MntTotal']/data.shape[0])*100,0)


# In[114]:


cluster_sizes


# In[116]:


plt.figure(figsize=(15, 8))
sns.boxplot(x='Cluster', y='Income', data=data, palette='viridis')
plt.title('Income by cluster')
plt.xlabel('Cluster')
plt.ylabel('Income')
plt.legend(title='Clusters')
plt.show()


# In[118]:


plt.figure(figsize=(15, 8))
sns.scatterplot(x='Income', y='MntTotal', data=data, hue = 'Cluster', palette='viridis')
plt.title('Income by cluster')
plt.xlabel('Income')
plt.ylabel('MntTotal')
plt.legend(title='Clusters')
plt.show()


# In[129]:


plt.figure(figsize=(15, 8))
sns.barplot(x='Cluster', y='In_relationship', data=data, palette='viridis')
plt.title('In_relationship by cluster')
plt.xlabel('Cluster')
plt.ylabel('In_relationship')
plt.show()


# In[ ]:





# In[131]:


data["Marital"]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




