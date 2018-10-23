import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing
import numpy as np


def read_data_file(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    classes = []
    signs = []
    for line in file:
        line = line.split(',')
        name = int(line.pop(0))
        classes.append(name)
        for i in range(len(line)):
            line[i] = float(line[i])
        signs.append(line)
    file.close()
    return np.array(signs), np.array(classes)


def maximum(array):
    number = 0
    m = array[0]
    for i in range(len(array)):
        if array[i] > m:
            m = array[i]
            number = i
    return m, number


def main():
    signs, classes = read_data_file('wine.data.txt')
    generator = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    qualities = []
    for i in range(1, 50):
        classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i)
        quality = sklearn.model_selection.cross_val_score(estimator=classifier, X=signs, y=classes, cv=generator)
        qualities.append(quality.mean())
    print(qualities)
    m, n = maximum(qualities)
    print(m, n)

    file = open('k_without_norming.txt', 'w', encoding='utf-8')
    file.write(str(n+1))
    file.close()

    file = open('value_without_norming.txt', 'w', encoding='utf-8')
    file.write(str(round(m, 2)))
    file.close()

    signs = sklearn.preprocessing.scale(signs)
    qualities = []
    for i in range(1, 50):
        classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i)
        quality = sklearn.model_selection.cross_val_score(estimator=classifier, X=signs, y=classes, cv=generator)
        qualities.append(quality.mean())
    print(qualities)
    m, n = maximum(qualities)
    print(m, n)

    file = open('k_with_norming.txt', 'w', encoding='utf-8')
    file.write(str(n + 1))
    file.close()

    file = open('value_with_norming.txt', 'w', encoding='utf-8')
    file.write(str(round(m, 2)))
    file.close()


main()
