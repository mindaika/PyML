import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from LearningCurvePlot import plot_learning_curve

# Read Yeast
trainFile = open('C://Users//Randall//OneDrive//Documents//Education//Grad School//Datasets//yeast.csv')
yeast = pd.read_csv(trainFile)
print("Training dataset shape: ", yeast.shape)

# Debugging
# print(df.columns.values)
# for val in df.columns.values:
#     print("\n", val, df[val].head(3))

# Encoding
enc = LabelEncoder()
yeast.loc[:, 'SequenceName'] = enc.fit_transform(yeast.loc[:, 'SequenceName'])
yeast.set_index(['SequenceName'], inplace=True)
yeast.loc[:, 'ClassDist'] = enc.fit_transform(yeast.loc[:, 'ClassDist'])


# Create Train/Test sets
train, test = train_test_split(yeast, test_size=0.2)

# Variable setup
X = yeast.drop('ClassDist', axis=1)
Y = yeast.loc[:, 'ClassDist']
X_train = train.drop('ClassDist', axis=1)
Y_train = train.loc[:, 'ClassDist']
X_test = test.drop('ClassDist', axis=1)
Y_test = test.loc[:, 'ClassDist']

# Create Classifier objects
# sklean Decision tree uses CART
# Finding the reference for that is left as an exercise for the reader
clf_gini = DecisionTreeClassifier(criterion="gini",
                                  random_state=100,
                                  max_depth=4,
                                  min_samples_split=29,
                                  min_samples_leaf=13)
clf_entropy = DecisionTreeClassifier(criterion="entropy",
                                     random_state=100,
                                     max_depth=10,
                                     min_samples_split=2,
                                     min_samples_leaf=16)
clf_knn = KNeighborsClassifier(n_neighbors=5,
                               n_jobs=4)
clf_adaB = RandomForestClassifier(n_estimators=16,
                                  min_samples_split=31,
                                  min_samples_leaf=9)
clf_MLP = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        learning_rate="adaptive",
                        random_state=1,
                        max_iter=500)
clf_SVC = SVC(C=68, kernel='rbf')
clf_dict = {'GINI_DTree': clf_gini,
            'Entropy_Tree': clf_entropy,
            'k_NN': clf_knn,
            'AdaBoosting': clf_adaB,
            'Perceptron_ANN': clf_MLP,
            'SVC': clf_SVC}

# Testing for best parameters
# for tester in range(50, 70):
#     sea = (tester/10)
#     clf_SVC = SVC(C=sea, kernel='rbf')
#     clf_SVC.fit(X_train, Y_train)
#     print("Unscaled SVC", tester, ": ", accuracy_score(Y_test, clf_SVC.predict(X_test)) * 100)

# Testing Part 2
# parameters = {'min_samples_leaf':range(2, 20)}
# clf = GridSearchCV(RandomForestClassifier(), parameters)
# clf.fit(X, Y)
# clf_adaB = clf.best_estimator_
# print(clf.best_score_, clf.best_params_)

# Fit Classifiers to Data and Print results
print("\n--Yeast Predictions--")

# Results Header
template = "{0:20}{1:15}{2:15}"
print(template.format("Classifier", "Accuracy(%)", "Runtime(s)"))
for key, value in clf_dict.items():
    start = time.time()
    value.fit(X_train, Y_train)
    end = time.time()
    print(template.format(key, ("%.2f" % (accuracy_score(Y_test, value.predict(X_test)) * 100)), "%.2f" % (end - start)))

# Plotter
for key, value in clf_dict.items():
    # Cross validation with n_splits iterations to get smoother mean test and train
    # score curves, each time with test_size data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    plt = plot_learning_curve(value, 'Yeast-' + key, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=1)
    plt.savefig('Yeast_' + key)

# # # K-Series
# # clf_gini.fit(K_train, Y_train)
# # print("\nGini: ", accuracy_score(Y_test, clf_gini.predict(K_test)) * 100)
# #
# # clf_entropy.fit(K_train, Y_train)
# # print("Entropy: ", accuracy_score(Y_test, clf_entropy.predict(K_test)) * 100)
# #
# # clf_adaB.fit(K_train, Y_train)
# # print("AdaBoost: ", accuracy_score(Y_test, clf_adaB.predict(K_test)) * 100)
# #
# # clf_knn.fit(K_train, Y_train)
# # print("kNN: ", accuracy_score(Y_test, clf_knn.predict(K_test)) * 100)
# #
# # clf_MLP.fit(K_train, Y_train)
# # print("MLP: ", accuracy_score(Y_test, clf_MLP.predict(K_test)) * 100)
# #
# # clf_SVC.fit(K_train, Y_train)
# # print("SVC: ", accuracy_score(Y_test, clf_SVC.predict(K_test)) * 100)
#
#
# # Data Visualization
# # title = "Learning Curves (Decision Tree)"

# # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# # plt = plot_learning_curve(clf_gini, title, X_train, Y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1)
#
# # title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # # SVC is more expensive so we do a lower number of CV iterations:
# # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# # estimator = SVC(gamma=0.001)
# # plt = plot_learning_curve(estimator, title, X, Y, (0.7, 1.01), cv=cv, n_jobs=1)