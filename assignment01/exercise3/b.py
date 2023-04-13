from sklearn import decomposition, datasets, svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from matplotlib import pyplot

import numpy as np

import ssl

# Have to create certificate to download faces
ssl._create_default_https_context = ssl._create_unverified_context

faces = datasets.fetch_olivetti_faces()
print(f"Faces shape: {faces.data.shape}")

fig = pyplot.figure(figsize=(8, 6))

# The faces are already scaled to the same size.
# Lets plot the first 20 of these faces.
for i in range(20):
    ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=pyplot.cm.bone)
    ax.set_title(i, fontsize='small', color='green')

pyplot.show()

# As usual, then lest split the dataset in a train and a test dataset.
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=0)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Choose our person that we want to find
y_test_person = y_test[10]

# Lets downscale the orginal pics with PCA n_components = Number of components to keep,
# Whitening = true can soemtimes improve the predictive accuracy of the downstream estimators
# by making their data respect some hard-wired assumptions.
pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)

# lets look at some of these faces. The socalled eigenfaces.
fig = pyplot.figure(figsize=(16, 6))

components_to_show = 12
components_per_row = 4
for i in range(components_to_show):
    ax = fig.add_subplot(components_to_show // components_per_row, components_per_row, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape), cmap=pyplot.cm.bone)

pyplot.show()

# With a PCA projection, the original pictures, train and test,
# can now be projected onto the PCA basis:
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"Train transformed shape: {X_train_pca.shape}")
print(f"Test transformed shape: {X_test_pca.shape}")

# We now use a SVM to make a classification
# kernel default = rbf, gamma = kernel coefficient
clf = svm.SVC(kernel="rbf", C=500, gamma=0.0001)
clf.fit(X_train_pca, y_train)

print("Classification report:")
print(classification_report(y_test, clf.predict(X_test_pca)))

# It is now time to evaluate how well this classification did.
# Lets look at the first 40 pics in the test set.
faces_to_show = 40
faces_per_row = 8

# Run our classifier on the test dataset
fig = pyplot.figure(figsize=(8, 6))
for i in range(faces_to_show):
    ax = fig.add_subplot(faces_to_show // faces_per_row, faces_per_row, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape), cmap=pyplot.cm.bone)

    y_pred = clf.predict(X_test_pca[i, np.newaxis])[0]
    color = ('blue' if y_pred == y_test[i] else 'red')

    ax.set_title(y_pred, fontsize='small', color=color)

pyplot.show()

# Run our classifier on the specific person test dataset
fig = pyplot.figure(figsize=(8, 6))
for i in range(faces_to_show):
    ax = fig.add_subplot(faces_to_show // faces_per_row, faces_per_row, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape), cmap=pyplot.cm.bone)

    y_pred = clf.predict(X_test_pca[i, np.newaxis])[0]
    color = ('blue' if y_pred == y_test_person else 'red')

    ax.set_title(y_pred, fontsize='small', color=color)

pyplot.show()
