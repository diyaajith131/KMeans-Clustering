import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load dataset (Customer Segmentation Example)
data = pd.read_csv("Mall_Customers.csv")

# Preprocessing
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Fit KMeans with optimal k (say k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)

# Add labels to data
data['Cluster'] = labels

# Visualization
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=labels, palette="viridis")
plt.title("Customer Segmentation with K-Means")
plt.show()

# Silhouette Score
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")
