from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from matplotlib import pyplot

from pandas import DataFrame

from typing_extensions import Literal

from exercise1.helpers import calc


def show_data():
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')

    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

    pyplot.show()


def use_svm(kernel: Literal["linear", "poly", "rbf", "sigmoid"]):
    classifier = SVC(kernel=kernel, C=100)
    classifier.fit(X_train, y_train)

    calc.plot_2d_separator(classifier, X_train, fill=True, alpha=.3)
    calc.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    pyplot.title(f"SVM ({kernel}) | Acc: {accuracy_score(y_test, classifier.predict(X_test))}")


if __name__ == '__main__':
    # generate 2d classification dataset
    X, y = make_circles(n_samples=100, noise=0.05)

    # scatter plot, dots colored by class value
    show_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    pyplot.figure(figsize=(10, 8))
    pyplot.subplot(2, 2, 1)
    use_svm("linear")
    pyplot.subplot(2, 2, 2)
    use_svm("poly")
    pyplot.subplot(2, 2, 3)
    use_svm("rbf")
    pyplot.subplot(2, 2, 4)
    use_svm("sigmoid")
    pyplot.show()
