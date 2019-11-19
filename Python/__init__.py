import inline as inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from Python import MyOwnMethods
from Python.MyOwnMethods import transform_features

data_train = pd.read_csv('../data/train.csv')
data_test = pd.read_csv('../data/test.csv')

# f, axes = plt.subplots(1, 2)
#
# sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train, ax=axes[0])
#
#
# sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train, ax=axes[1],
#               palette={"male": "blue", "female": "pink"},
#               markers=["*", "o"], linestyles=["-", "--"])
# plt.show()


data_train = transform_features(data_train)
data_test = transform_features(data_test)
data_train.head()

sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train)

data_train, data_test = MyOwnMethods.encode_features(data_train, data_test)
print(data_train.head())

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

# Choose the type of classifier.
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt', 'auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
              }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data.
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

# k-fold accuracy score determining
def run_kfold(clf):
    kf = KFold(n_splits=10)
    outcomes = []
    fold = 0
    for trainindex, testindex in kf.split(X_all):
        fold += 1
        Xtrain, Xtest = X_all.values[trainindex], X_all.values[testindex]
        ytrain, ytest = y_all.values[trainindex], y_all.values[testindex]
        clf.fit(Xtrain, ytrain)
        predictions = clf.predict(Xtest)
        accuracy = accuracy_score(ytest, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    meanoutcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(meanoutcome))

run_kfold(clf)



ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)
print(output)


