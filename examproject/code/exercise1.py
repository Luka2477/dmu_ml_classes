import random
import sys

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from typing import TypedDict, Any

# Download the Breast Cancer Dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Divide the data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

print("Train dataset shape")
print(X_train.shape)
print(y_train.shape)
print()
print("Test dataset shape")
print(X_test.shape)
print(y_test.shape)
print()

# Exercise 1 : Improve precision on training set.
# Make relevant changes to the code below.

# Decision Tree Classifier* using all the features of the data. Model tested on the test data
tree = DecisionTreeClassifier(criterion='gini',
                              max_depth=9,
                              random_state=0)
tree.fit(X_train, y_train)
plot_tree(tree,
          feature_names=cancer.feature_names,
          fontsize=8)

# Generate confusion matrix on train dataset
y_train_true = y_train
y_train_pred = tree.predict(X_train)
print("Precision on training set")
print(confusion_matrix(y_train_true, y_train_pred))
print()

# Generate confusion matrix on test dataset
y_tes_true = y_test
y_test_pred = tree.predict(X_test)
print("Precision on test set")
print(confusion_matrix(y_tes_true, y_test_pred))
print()

print("Train Set Accuracy : ", accuracy_score(y_train, y_train_pred))
print("Test Set Accuracy  : ", accuracy_score(y_test, y_test_pred))
print()


# Gini and entropy statistics
class StatsInner(TypedDict):
    gini: Any
    entropy: Any


class StatsOuter(TypedDict):
    train: StatsInner
    test: StatsInner


def gen_stats(depth: int) -> StatsOuter:
    tree_gin = DecisionTreeClassifier(criterion='gini',
                                      max_depth=depth,
                                      random_state=0)
    tree_gin.fit(X_train, y_train)

    tree_ent = DecisionTreeClassifier(criterion='entropy',
                                      max_depth=depth,
                                      random_state=0)
    tree_ent.fit(X_train, y_train)

    return {
        "train": {
            "gini": tree_gin.predict(X_train),
            "entropy": tree_ent.predict(X_train),
        },
        "test": {
            "gini": tree_gin.predict(X_test),
            "entropy": tree_ent.predict(X_test),
        },
    }


def print_stats(depth: int, stats: StatsOuter) -> None:
    print(f"DEPTH = {depth}")
    print("\tGINI : ")
    print("\t\tTrain Set Accuracy : ", accuracy_score(y_train, stats["train"]["gini"]))
    print("\t\tTest Set Accuracy  : ", accuracy_score(y_test, stats["test"]["gini"]))
    print("\tENTROPY : ")
    print("\t\tTrain Set Accuracy : ", accuracy_score(y_train, stats["train"]["entropy"]))
    print("\t\tTest Set Accuracy  : ", accuracy_score(y_test, stats["test"]["entropy"]))
    print()


# Exercise 2
# Make similar code as depth 1 and 6 for depth 2,3 and 7
depth = 1
print_stats(depth, gen_stats(depth))

depth = 2
print_stats(depth, gen_stats(depth))

depth = 3
print_stats(depth, gen_stats(depth))

depth = 4
print_stats(depth, gen_stats(depth))

depth = 5
print_stats(depth, gen_stats(depth))

depth = 6
print_stats(depth, gen_stats(depth))

depth = 7
print_stats(depth, gen_stats(depth))

# Exercise 3
# For which tree height (1 -7) do we find the highest accuracy for test set ?
# Depth 3 for both "gini" and "entropy"

# Exercise 4
# Improve the accuracy of the SVM classifier.
svm = SVC(kernel="rbf", C=45000.0, random_state=0)
svm.fit(X_train, y_train)

Y_train_pred = svm.predict(X_train)
print("SVM Train Set Accuracy : ", accuracy_score(y_train, Y_train_pred))

Y_test_pred = svm.predict(X_test)
print("SVM Test Set Accuracy  : ", accuracy_score(y_test, Y_test_pred))
print()

# Exercise 5
# Improve the accuracy of the Logistics Regression classifier.
lr = LogisticRegression(penalty='l2', solver="lbfgs", C=225.0, random_state=0, max_iter=100000)
# lr = LogisticRegression(penalty='l2', solver="newton-cg", C=135.0, random_state=0, max_iter=100000)
lr.fit(X_train, y_train)

Y_train_pred = lr.predict(X_train)
print("LR Train Set Accuracy : ", accuracy_score(y_train, Y_train_pred))

Y_test_pred = lr.predict(X_test)
print("LR Test Set Accuracy  : ", accuracy_score(y_test, Y_test_pred))
print()

# Exercise 6
# Make a neural net classifier.

# While loop that randomly generates node layers and sizes
# Tries to find the best possible solution using an MLPC

# train_max = 0
# test_max = 0
# while True:
#     mlp = MLPClassifier(solver="lbfgs",
#                         hidden_layer_sizes=[random.randint(2, 250) for _ in range(random.randint(1, 3))],
#                         max_iter=sys.maxsize)
#     mlp.fit(X_train, y_train)
#
#     Y_train_pred = mlp.predict(X_train)
#     Y_test_pred = mlp.predict(X_test)
#     Y_train_acc = accuracy_score(y_train, Y_train_pred)
#     Y_test_acc = accuracy_score(y_test, Y_test_pred)
#
#     if Y_train_acc > train_max:
#         better = True
#         train_max = Y_train_acc
#         print(f"BEST TRAIN   |  {train_max:.16f} : {mlp.hidden_layer_sizes}")
#     else:
#         print(f"WORSE TRAIN  |  {Y_train_acc:.16f} : {mlp.hidden_layer_sizes}")
#
#     if Y_test_acc > test_max:
#         better = True
#         test_max = Y_test_acc
#         print(f"BEST TEST    |  {test_max:.16f} : {mlp.hidden_layer_sizes}")
#     else:
#         print(f"WORSE TEST   |  {Y_test_acc:.16f} : {mlp.hidden_layer_sizes}")

# BEST TRAIN   |  0.9978021978021978 : [193, 193, 49]
# BEST TEST    |  0.9736842105263158 : [166, 90]

mlp = MLPClassifier(solver="lbfgs", hidden_layer_sizes=[215, 210, 164], max_iter=sys.maxsize)
mlp.fit(X_train, y_train)

Y_train_pred = mlp.predict(X_train)
print("MLP Train Set Accuracy : ", accuracy_score(y_train, Y_train_pred))

Y_test_pred = mlp.predict(X_test)
print("MLP Test Set Accuracy  : ", accuracy_score(y_test, Y_test_pred))
print()

# Exercise 7
# In what way do you achieve the highest accuracy?
