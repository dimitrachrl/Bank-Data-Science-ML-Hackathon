import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler,RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

dataframe=pd.read_csv('the_users.csv')

user_id=dataframe['user_id']


features=[feature for feature in dataframe.columns if feature not in ['days_from_last_trans','sum_amount_of_transactions',
                                                                       'count_of_transactions']]






for feature in features:
    dataframe.drop(feature,axis=1,inplace=True)
    

dataframe['days_from_last_trans']=dataframe['days_from_last_trans']-365

scaler=MinMaxScaler()
    
X=scaler.fit_transform(dataframe)




    
sil = []
kmax = 8

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(dataframe)
  clusters = kmeans.predict(dataframe)
  sil.append(silhouette_score(dataframe, clusters, metric = 'euclidean'))



kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)
kmeanspred=kmeans.fit_predict(X)



'''

plt.scatter(dataframe.iloc[:,0], dataframe.iloc[:,1] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.title("Clustered users")
plt.xlabel('days_from_first_trans')
plt.ylabel('days_from_last_trans')
plt.show()


plt.scatter(dataframe.iloc[:,0], dataframe.iloc[:,2] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.title("Clustered users")
plt.xlabel('days_from_first_trans')
plt.ylabel('sum_amount_of_transactions')
plt.show()


plt.scatter(dataframe.iloc[:,3], dataframe.iloc[:,0] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.title("Clustered users")
plt.xlabel('count_of_transactions')
plt.ylabel('days_from_first_trans')
plt.show()



plt.scatter(dataframe.iloc[:,1], dataframe.iloc[:,2] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.title("Clustered users")
plt.xlabel('days_from_last_trans')
plt.ylabel('sum_amount_of_transactions')
plt.show()



plt.scatter(dataframe.iloc[:,1], dataframe.iloc[:,3] ,c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.title("Clustered users")
plt.xlabel('days_from_last_trans')
plt.ylabel('count_of_transactions')
plt.show()


plt.scatter(dataframe.iloc[:,2], dataframe.iloc[:,3] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.title("Clustered users")
plt.xlabel('sum_amount_of_transactions')
plt.ylabel('count_of_transactions')
plt.show()

'''


##################################################################################################################



plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='Spectral',  alpha=0.8, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='orange',label='Centroids', edgecolors='black')
plt.title("Clustered users")
plt.xlabel('Days Since Last Transaction')
plt.ylabel('Total Amount')
plt.show()



plt.scatter(X[:,0], X[:,2] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,2],s=300,c='orange',label='Centroids', edgecolors='black')
plt.title("Clustered users")
plt.xlabel('Days Since Last Transaction')
plt.ylabel('Count of Transactions')
plt.show()

plt.scatter(X[:,0], X[:,3] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,3],s=300,c='orange',label='Centroids', edgecolors='black')
plt.title("Clustered users")
plt.xlabel('Days Since First Transaction')
plt.ylabel('Count of Transactions')
plt.show()



plt.scatter(X[:,2], X[:,1] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=300,c='orange',label='Centroids', edgecolors='black')
plt.title("Clustered users")
plt.xlabel('Days Since Last Transaction')
plt.ylabel('Τotal Amount')
plt.show()


plt.scatter(X[:,1], X[:,3] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,3],s=300,c='orange',label='Centroids', edgecolors='black')
plt.title("Clustered users")
plt.xlabel('Days Since Last Transaction')
plt.ylabel('Count of Transactions')
plt.show()




plt.scatter(X[:,2], X[:,3] , c=kmeans.labels_, cmap='Spectral', alpha=0.8, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:,2],kmeans.cluster_centers_[:,3],s=300,c='orange',label='Centroids', edgecolors='black')
plt.title("Clustered users")
plt.xlabel('Τotal Amount')
plt.ylabel('Count of Transactions')
plt.show()




centroids = scaler.inverse_transform(np.array(kmeans.cluster_centers_))
df=pd.DataFrame(centroids, columns=dataframe.columns)




X_final=pd.DataFrame(X, columns=['days_from_last_trans','sum_amount_of_transactions','count_of_transactions'])

kmeanspred=pd.Series(kmeanspred).rename('Group')


user_modeling=pd.concat([X_final,kmeanspred],axis=1)


grouped_users=pd.concat([user_id,kmeanspred],axis=1)


user_modeling.to_csv('user_modeling.csv',index=False)

grouped_users['Group'].value_counts()










df.to_csv('examples.csv', index=False)









