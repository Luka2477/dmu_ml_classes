# -*- coding: utf-8 -*-
"""
Created on Mon September 20 16:14:02 2022
@author: sila
"""
#Import scikit-learn dataset library
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
x, y = make_classification(n_samples=5000, n_features=10,
                           n_classes=3,
                           n_clusters_per_class=1)
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)
# lsvc = LinearSVC(C=1, max_iter=10000000)
lsvc = SVC(kernel='rbf', random_state=1, gamma=1, C=10)
print(lsvc)
lsvc.fit(xtrain, ytrain)
score = lsvc.score(xtrain, ytrain)
print("Score: ", score)
cv_scores = cross_val_score(lsvc, xtrain, ytrain, cv=10)
print("CV average score: %.2f" % cv_scores.mean())
ypred = lsvc.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
cr = classification_report(ytest, ypred)
print(cr)
# exit()
# Iris dataset classification
print("Iris dataset classification with SVC")
iris = load_iris()
x, y = iris.data, iris.target
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)
lsvc = LinearSVC(C=0.5, max_iter=10000000)
print(lsvc)
lsvc.fit(xtrain, ytrain)
score = lsvc.score(xtrain, ytrain)
print("Score: ", score)
cv_scores = cross_val_score(lsvc, xtrain, ytrain, cv=10)
print("CV average score: %.2f" % cv_scores.mean())
ypred = lsvc.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
cr = classification_report(ytest, ypred)
print(cr)