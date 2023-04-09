from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from matplotlib import pyplot

from pandas import DataFrame

from exercise1.helpers import moonscalc

if __name__ == '__main__':
    # generate 2d classification dataset
    X, y = make_moons(n_samples=100, noise=0.1)

    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')

    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

    pyplot.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[50, 25, 10], alpha=0.01)
    mlp.fit(X_train, y_train)

    moonscalc.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    moonscalc.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    pyplot.show()
