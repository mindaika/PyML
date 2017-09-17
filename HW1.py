from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as pp
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np
import LearningCurvePlot as pt
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_digits




# Read Training set
trainFile = open('C://Users//Randall//OneDrive//Documents//Education//Grad School//Datasets//adult_edit.csv')
train = pd.read_csv(trainFile)
train = train.fillna('')
train.index.name = 'Row'
# print("Training dataset length: ", len(train))
print("Training dataset shape: ", train.shape)

# Read Test set
testFile = open('C://Users//Randall//OneDrive//Documents//Education//Grad School//Datasets//adultCV.csv')
test = pd.read_csv(testFile)
test = test.fillna('')
test.index.name = 'Row'
# print("Training dataset length: ", len(test))
print("Training dataset shape: ", test.shape)

# Slice data into testers
X_train = train.iloc[:, 0:13]
Y_train = train.iloc[:, 14]
X_test = test.iloc[:, 0:13]
Y_test = test.iloc[:, 14]

print(X_train.head.columns.values)

# Lightly munge data
# train_obj = train.select_dtypes(include=['object'])
# test_obj = test.select_dtypes(include=['object'])
# train[train_obj.columns] = train_obj.apply(lambda x: x.str.lstrip())

# Encode data, since sklearn only works with numbers
Xtrain_dummy = pd.get_dummies(X_train)
Ytrain_dummy = Y_train["income_bin"].apply(lambda income: 0 if income == ">50k" else 1)
Xtest_dummy = pd.get_dummies(X_test)
Ytest_dummy = Y_test["income_bin"].apply(lambda income: 0 if income == ">50k" else 1)

print("XTest_Dummy shape: ", Xtest_dummy.shape)
print("YTest_Dummy shape: ", Ytest_dummy.shape)
print("XTrain_Dummy shape: ", Xtrain_dummy.shape)
print("YTrain_Dummy shape: ", Ytrain_dummy.shape)



# Create DecisionTreeClassifier
# sklean Decision tree uses CART
# Finding the reference for that is left as an exercise for the reader
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                  max_depth=5, min_samples_leaf=50)
clf_gini.fit(Xtrain_dummy, Ytrain_dummy)

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                     max_depth=3, min_samples_leaf=50)
clf_entropy.fit(Xtrain_dummy, Ytrain_dummy)

print(accuracy_score(Ytest_dummy, clf_gini.predict(Xtest_dummy)) * 100)
print(accuracy_score(Ytest_dummy, clf_entropy.predict(Xtest_dummy)) * 100)


# Data visualization

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
pt.plot_learning_curve(estimator, title, Xtrain_dummy, Ytrain_dummy, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
pt.plot_learning_curve(estimator, title, Xtrain_dummy, Ytrain_dummy, (0.7, 1.01), cv=cv, n_jobs=1)

plt.show()


# feature_names = list(Xtrain_dummy)
# dot_data = tree.export_graphviz(clf_gini, out_file=None, feature_names=feature_names)
# graph = graphviz.Source(dot_data)
# graph.render("adult")
