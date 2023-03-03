import numpy as np
import pandas
from sklearn import datasets
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression

# Load dataset
url = "/Users/lukasknudsen/Documents/dmu/ml/dmu_ml_classes/day04_logistic_regression/task5/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# We will now try to take a closer look at (only) the petal-width, and see if we can make a
# classification based on that.
array = dataset.values
X = array[:, 3]  # petal width
Y = array[:, 4]  # classification

# Make y into numbers
a_enc = pandas.factorize(Y)
yvalues = a_enc[0]

y = []

for i in yvalues:
    if i == 2:
        y = np.append(y, [1])
    else:
        y = np.append(y, [0])

    # l2 is default,
# Intuitively, the model will be adjusted to minimize single outlier case,
# at the expense of many other common examples
log_reg = LogisticRegression(penalty="l2")
# Shape, gives a new shape to an array without changing its data. In order to use algorithm
log_reg.fit(X.reshape(-1, 1), y)

plt.plot(X, y, "b.")

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginca")
plt.xlabel("Petal width", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.show()

# Next, it would be interesting to see a plot for both petal-width and petal-length.
iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]  # petal length, petal width
yy = iris["target"]

y = []
for i in yy:
    if i == 2:
        y = np.append(y, [1])
    else:
        y = np.append(y, [0])

    # model = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5)
model = LogisticRegression(C=1000)
model.fit(X, y)

plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "y.", label="Virginica")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "b.", label="Setosa or versicolor")

plt.legend(loc="upper left", fontsize=14)

plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)

xx = np.linspace(1, 6)
yy = -  (model.coef_[0, 0] / model.coef_[0, 1]) * xx - (model.intercept_[0] / model.coef_[0, 1])

plt.plot(xx, yy, 'k-')
plt.show()
