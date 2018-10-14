import pandas
import copy
import numpy as np


def men_and_women_count(data):
    series = data['Sex'].value_counts()
    print(str(series['male']) + ' ' + str(series['female']))
    file = open('men_and_women_count.txt', 'w', encoding='utf-8')
    file.write(str(series[0]) + ' ' + str(series[1]))
    file.close()


def survivors_count(data):
    series = data['Survived'].value_counts()
    print(round(series[1] / data.shape[0] * 100, 2))
    file = open('survivors_count.txt', 'w', encoding='utf-8')
    file.write(str(round(series[1] / data.shape[0] * 100, 2)))
    file.close()


def first_class_count(data):
    series = data['Pclass'].value_counts()
    print(round(series[1] / data.shape[0] * 100, 2))
    file = open('first_class_count.txt', 'w', encoding='utf-8')
    file.write(str(round(series[1] / data.shape[0] * 100, 2)))
    file.close()


def average_age(data):
    mean = data['Age'].mean()
    median = data['Age'].median()
    print(str(round(mean, 2)) + ' ' + str(median))
    file = open('average_age.txt', 'w', encoding='utf-8')
    file.write(str(round(mean, 2)) + ' ' + str(median))
    file.close()


def correlation(data):
    K = np.corrcoef(data['SibSp'].values, data['Parch'].values)
    print(round(K[0, 1], 2))
    file = open('correlation.txt', 'w', encoding='utf-8')
    file.write(str(round(K[0, 1], 2)))
    file.close()


def get_name(full_name):
    if 'Mrs.' in full_name and '(' in full_name:
        return full_name.split('(')[1].split()[0]
    else:
        return full_name.split('. ')[1].split()[0]


def popular_women_name(data):
    women_names = data[data.Sex == 'female'].Name.apply(get_name)
    name = women_names.value_counts().index[0]
    print(name)
    file = open('popular_women_name.txt', 'w', encoding='utf-8')
    file.write(name)
    file.close()


def main():
    data = pandas.read_csv('titanic.csv', index_col='PassengerId')
    print(data.shape)
    print(data.axes)
    men_and_women_count(data)
    survivors_count(data)
    first_class_count(data)
    average_age(data)
    correlation(data)
    popular_women_name(data)


main()
