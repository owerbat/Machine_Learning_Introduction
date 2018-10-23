import sklearn.neighbors
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import numpy as np


def maximum_index(array):
    number = array[0][0]
    m = array[0][1]
    for el in array:
        if el[1] > m:
            m = el[1]
            number = el[0]
    return number


def main():
    dataset = sklearn.datasets.load_boston()
    dataset.data = sklearn.preprocessing.scale(dataset.data)
    generator = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    qualities = []
    for i in np.linspace(1, 10, num=200):
        regressor = sklearn.neighbors.KNeighborsRegressor(5, weights='distance', p=i, metric='minkowski')
        quality = sklearn.model_selection.cross_val_score(estimator=regressor, X=dataset.data,
                                                          y=dataset.target, cv=generator)
        qualities.append([i, quality.mean()])
    print(qualities)
    p = maximum_index(qualities)
    print(p)

    file = open('p_value_in_metric.txt', 'w', encoding='utf-8')
    file.write(str(round(p, 1)))
    file.close()


main()
