import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df2 = pd.read_csv("final.csv") 

from sklearn.cluster import KMeans
import pandas as pd

print(df2)

X = df2.iloc[:,[3,4]].values 

wcss= [ ] 
for i in range(1,11): 
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42) 
  kmeans.fit(X) 

  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel('Number of Clusters')
plt.show()