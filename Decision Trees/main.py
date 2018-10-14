import pandas
from sklearn.tree import DecisionTreeClassifier


def main():
    data = pandas.read_csv('titanic.csv', index_col='PassengerId')
    data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
    data = data.dropna()
    data.Sex = data.Sex.replace(to_replace=['male', 'female'], value=[0, 1])
    # data.Sex = data.Sex.apply(lambda sex: 0 if sex == 'male' else 1)
    #print(data)

    signs = ['Pclass', 'Fare', 'Age', 'Sex']
    X = data[signs].values
    y = data.Survived.values
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(X, y)
    importances = clf.feature_importances_

    table = [[importances[i], signs[i]] for i in range(len(signs))]
    table.sort(reverse=True)
    print(table)
    print(table[0][1] + ' ' + table[1][1])
    file = open('decision_trees.txt', 'w', encoding='utf-8')
    file.write(table[0][1] + ' ' + table[1][1])


main()
