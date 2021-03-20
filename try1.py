import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

creditcard_db = pd.read_csv("CC GENERAL.CSV")

creditcard_db  # to see CSV file in detail
creditcard_db.info()   #to check file info
creditcard_db.describe()  #to get statistcal info

#to check missing elements
sns.heatmap(creditcard_db.isnull(), yticklabels=False, cbar=False, cmap="Blues" )
#summumary of missing elements
creditcard_db.isnull().sum()
#filling those missing elements with mean
creditcard_db.loc[(creditcard_db['MINIMUM_PAYMENTS'].isnull()==True), 'MINIMUM_PAYMENTS'] = creditcard_db['MINIMUM_PAYMENTS'].mean()
creditcard_db.loc[(creditcard_db['CREDIT_LIMIT'].isnull()==True), 'CREDIT_LIMIT'] = creditcard_db['CREDIT_LIMIT'].mean()
creditcard_db.isnull().sum()
creditcard_db.duplicated().sum()
sns.heatmap(creditcard_db.isnull(), yticklabels=False, cbar=False, cmap="Blues" )
#dropping irrelavant columns
creditcard_db.drop('CUST_ID', axis=1, inplace=True)
creditcard_db

creditcard_db.columns
n = len(creditcard_db.columns)

#for visualisation
plt.figure(figsize=(10,50))
for i in range(len(creditcard_db.columns)):
    plt.subplot(17,1,i+1)
    sns.distplot(creditcard_db[creditcard_db.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE", "bw"=0}, hist_kws={"color": "g"})
    plt.title(creditcard_db.columns[i])    
plt.tight_layout()

#obtaining corealtion matrix
f, ax = plt.subplots(figsize=(20,10))
sns.heatmap(creditcard_db.corr() , annot=True)
plt.show()

#scaling the data
scaler = StandardScaler()
creditcard_db_scaled= scaler.fit_transform(creditcard_db)

#elbow method
score = []
for i in range(1,20):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(creditcard_db_scaled)
    score.append(kmeans.inertia_)
    
plt.plot(score, "bx-")
plt.xlabel("no. of clusters")
plt.ylabel("WCSS scores")
plt.show()    

#applying KMeans
kmean = KMeans(n_clusters= 8)
kmean.fit(creditcard_db_scaled)
labels = kmean.labels_

kmean.cluster_centers_.shape

cluster_centers = pd.DataFrame(data = kmean.cluster_centers_, columns= [creditcard_db.columns])
cluster_centers #with scalar values
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data= cluster_centers, columns = [creditcard_db.columns])
cluster_centers #with real values (inverse scalar)

labels.shape #total no of labels given to each
labels.max() #max label no.
labels.min() #min label no.
#adding cluster column/labels to creditcard_db
creditcard_db_cluster = pd.concat([creditcard_db, pd.DataFrame({"cluster no.": labels})], axis = 1)
creditcard_db_cluster.head() #shows intital 5 

#PCA
pca = PCA(n_components=2)
principal_comp= pca.fit_transform(creditcard_db_scaled)
pca_db = pd.DataFrame(data= principal_comp, columns=["pca1", 'pca2'])
pca_db = pd.concat([pca_db,pd.DataFrame({"cluster no.": labels})], axis = 1)
pca_db

plt.figure(figsize=(10,10))
ax = sns.scatterplot(x='pca1', y='pca2', hue = "cluster no.", data = pca_db, palette=['red', 'yellow', 'blue', 'green', 'purple', 'orange', 'grey', 'black'])
plt.show()