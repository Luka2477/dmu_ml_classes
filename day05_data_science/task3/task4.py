from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm

iris_dataset = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

logreg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, max_iter=1000000, multi_class='ovr', n_jobs=1,
                            penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                            verbose=0, warm_start=False)

logreg.fit(X_train, Y_train)
print("Training scores: {:.2f}".format(logreg.score(X_train, Y_train)))
print("Test scores: {:.2f}".format(logreg.score(X_test,Y_test)))
print()

clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)

print("Training scores: {:.2f}".format(clf.score(X_train, Y_train)))
print("Test scores: {:.2f}".format(clf.score(X_test, Y_test)))