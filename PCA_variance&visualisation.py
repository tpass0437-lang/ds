# explained variance 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("C:\dataset\iris-write-from-docker.csv")  
# 🔴 CHANGE FILE PATH

X = df.select_dtypes(include='number')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)


# visualisation of PCA projection

import matplotlib.pyplot as plt

pca_2 = PCA(n_components=2)  
# 🔴 CHANGE NUMBER OF COMPONENTS IF REQUIRED

X_pca_2 = pca_2.fit_transform(X_scaled)

plt.scatter(X_pca_2[:,0], X_pca_2[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()
