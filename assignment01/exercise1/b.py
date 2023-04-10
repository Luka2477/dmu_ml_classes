from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from matplotlib import pyplot

from pandas import DataFrame

from numpy import array

from exercise1.helpers import calc


def show_data():
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')

    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

    pyplot.show()


def use_mlp():
    classifier = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[50, 25, 10], alpha=0.01)
    classifier.fit(X_train, y_train)

    calc.plot_2d_separator(classifier, X_train, fill=True, alpha=.3)
    calc.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    pyplot.title("Multi-Layer Perceptron")
    pyplot.show()


def use_logreg():
    classifier = linear_model.LogisticRegression(C=1)
    classifier.fit(X_train, y_train)

    calc.plot_2d_separator(classifier, X_train, fill=True, alpha=.3)
    calc.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    pyplot.title("Logistic Regression")
    pyplot.show()


def use_kmeans():
    classifier = KMeans(n_clusters=2, random_state=0, n_init="auto")
    classifier.fit(X_train)

    colormap = array(['red', 'lime', 'black', 'blue', 'yellow', 'green', 'red'])
    pyplot.scatter(X_train[:, 0], X_train[:, 1], c=colormap[classifier.labels_])

    pyplot.title("K-Means")
    pyplot.show()


def use_dectree():
    classifier = DecisionTreeClassifier(max_depth=5)
    classifier.fit(X_train, y_train)

    calc.plot_2d_separator(classifier, X_train, fill=True, alpha=.3)
    calc.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    pyplot.title("Decision Tree")
    pyplot.show()


def use_randfor():
    classifier = RandomForestClassifier(max_depth=5, n_estimators=25)
    classifier.fit(X_train, y_train)

    calc.plot_2d_separator(classifier, X_train, fill=True, alpha=.3)
    calc.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    pyplot.title("Random Forest")
    pyplot.show()


def use_svm():
    classifier = SVC(C=100)
    classifier.fit(X_train, y_train)

    calc.plot_2d_separator(classifier, X_train, fill=True, alpha=.3)
    calc.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    pyplot.title("Support Vector Machine")
    pyplot.show()


if __name__ == '__main__':
    # generate 2d classification dataset
    X, y = make_moons(n_samples=100, noise=0.1)

    # scatter plot, dots colored by class value
    show_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    use_mlp()
    use_logreg()
    use_kmeans()
    use_dectree()
    use_randfor()
    use_svm()
