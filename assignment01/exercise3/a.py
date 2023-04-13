from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from matplotlib import pyplot

import numpy as np

iris_df = datasets.load_iris()
pca = PCA(2)

X, y = iris_df.data, iris_df.target
X_proj = pca.fit_transform(X)

pyplot.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
pyplot.show()

# Principal axes in feature space, representing the directions of maximum variance in the data.
# The components are sorted by explained_variance_.
# pca.components_ has the meaning of each principal component, essentially how it was derived
# checking shape tells us it has 2 rows, one for each principal component and 4 columns,
# proportion of each of the 4 features for each row
print(f"PCA PCs: {pca.components_}")
print(f"PCA PC shape: {pca.components_.shape}")

# this tells us the extent to which each component explains the original dataset.
# so the 1st component is able to explain ~92% of X and the second only about 5.3%
# Together they can explain about 97.3% of the variance of X
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print()

# Set the size of the plot
pyplot.figure(figsize=(10, 4))

# create color map
colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow', 'green', 'red'])

# running kmeans clustering into two
k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X_proj)

# printing our cluster centers - there will be 2 of them.
print(f"K-means clusters centers: {kmeans.cluster_centers_}")
print()

# print our accuracy score using the k-means algorithm
labels = kmeans.labels_
print(f"Accuracy score: {accuracy_score(labels, y)}")
flipped = [val if not val else 1 if val == 2 else 2 for val in labels]
print(f"Accuracy score flipped: {accuracy_score(flipped, y)}")

# Plot the Original Classifications
pyplot.subplot(1, 3, 1)
pyplot.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[y], s=40)
pyplot.title('Real Classification')

# Plot the Models Classifications
# the random state is optionly, here it is specified so we get deterministic clusters.
# this will contain the labels for our predicted clusters (either 0 or 1)
pyplot.subplot(1, 3, 2)
pyplot.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[labels], s=40)
pyplot.title('K Mean Classification')

# Plot the flipped Models Classification
pyplot.subplot(1, 3, 3)
pyplot.scatter(X_proj[:, 0], X_proj[:, 1], c=colormap[flipped], s=40)
pyplot.title('K Mean Classification Flipped')

pyplot.show()

