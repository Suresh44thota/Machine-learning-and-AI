import pandas as pd 
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
df= pd.read_csv('data.csv')
# Partion the dependent and independent variables
y = df.Class
X= df.drop(['Class'], axis = 1)
# obtainig the optimal value of k
from sklearn.cluster import KMeans

clusters = []

for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)
    
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# Annotate arrow
ax.annotate('Possible Elbow Point', xy=(3, 140000), xytext=(3, 50000), xycoords='data',          
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible Elbow Point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',          
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

plt.show()
algorithm = (KMeans(n_clusters = 6))
algorithm.fit(X)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_
print(labels1)
print(centroids1)
# caluculating adjusted_rand_score, v_measure_score,homogeneity_score,completeness_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score

print('Adjusted rand score %.6f' % adjusted_rand_score(y,labels1))
print("V_measure score %.6f" % v_measure_score(y,labels1))
print("Homogeneity score %.6f" % homogeneity_score(y,labels1))
print("Completeness score %.6f" % completeness_score(y,labels1))

# fine tuning
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X)
labels1 = algorithm.labels_
print('Adjusted rand score %.6f' % adjusted_rand_score(y,labels1))
print("V_measure score %.6f" % v_measure_score(y,labels1))
print("Homogeneity score %.6f" % homogeneity_score(y,labels1))
print("Completeness score %.6f" % completeness_score(y,labels1))

# Agglomerative Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering 

agglom = AgglomerativeClustering(n_clusters=6, linkage='average',).fit(X)

X['Labels'] = agglom.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Alcohol'], X['Magnesium'], hue=X['Labels'], 
                palette=sns.color_palette('hls', 6))
plt.title('Agglomerative with 4 Clusters')
plt.show()
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
# caluculating adjusted_rand_score, v_measure_score,homogeneity_score,completeness_score

print('Adjusted rand score %.6f' % adjusted_rand_score(y,X.Labels))
print("V_measure score %.6f" % v_measure_score(y,X.Labels))
print("Homogeneity score %.6f" % homogeneity_score(y,X.Labels))
print("Completeness score %.6f" % completeness_score(y,X.Labels))


# Dendogram
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 

dist = distance_matrix(X, X)
print(dist)
#Z = hierarchy.linkage(dist, 'complete')
Z=hierarchy.linkage(dist,method="ward")
plt.figure(figsize=(15,10))
hierarchy.dendrogram(Z,leaf_rotation=90,p=5,color_threshold=20,leaf_font_size=10,truncate_mode='level')
plt.axhline(y=125, color='r', linestyle='--')
plt.show()
#plt.figure(figsize=(18, 50))
#dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')
# Fine tuning of the parameters
agglom = AgglomerativeClustering(n_clusters=6, linkage='complete',).fit(X)
# caluculating adjusted_rand_score, v_measure_score,homogeneity_score,completeness_score

X['Labels'] = agglom.labels_
print('Adjusted rand score %.6f' % adjusted_rand_score(y,X.Labels))
print("V_measure score %.6f" % v_measure_score(y,X.Labels))
print("Homogeneity score %.6f" % homogeneity_score(y,X.Labels))
print("Completeness score %.6f" % completeness_score(y,X.Labels))

