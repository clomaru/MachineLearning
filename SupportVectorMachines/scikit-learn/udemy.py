import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data

Y = iris.target

print(iris.DESCR)
from sklearn.svm import SVC
model = SVC()
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=3)
# モデルをトレーニングします。
model.fit(X_train,Y_train)

from sklearn import metrics

predicted = model.predict(X_test)
expected = Y_test

print(metrics.accuracy_score(expected,predicted))
