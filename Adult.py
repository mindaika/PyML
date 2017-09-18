import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from LearningCurvePlot import plot_learning_curve

# Read Training set
trainFile = open('C://Users//Randall//OneDrive//Documents//Education//Grad School//Datasets//adult_edit.csv')
train = pd.read_csv(trainFile)
train = train.fillna('')
train.index.name = 'Row'
print("Train dataset shape: ", train.shape)

# Read Test set
testFile = open('C://Users//Randall//OneDrive//Documents//Education//Grad School//Datasets//adultCV.csv')
test = pd.read_csv(testFile)
test = test.fillna('')
test.index.name = 'Row'
print("Test dataset shape: ", test.shape)

# Concatenate sets for pre-processing
df = pd.concat([train, test], keys=["train", "test"])

# Booleate (real word) sex, income_bin
# Sex: Female=0, Male=1
# Income_bin: <=50K=0, >50=1
df['income_bin'] = df['income_bin'].apply(lambda income: 0 if income == '<=50K' else 1)
df['sex'] = df['sex'].apply(lambda sex: 0 if sex == "Male" else 1)

# Encode dataset features
labels_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
df_with_dumbo = pd.get_dummies(df, labels_to_encode)

# Create Train/Test sets with dummies
X = df_with_dumbo.drop('income_bin', axis=1)
Y = df_with_dumbo['income_bin']

X_train = X.loc['train']
Y_train = Y.loc['train']
X_test = X.loc['test']
Y_test = Y.loc['test']

# Feature Selection Comparison
selector = SelectKBest(chi2, k=2)
K_train = selector.fit_transform(X_train, Y_train)
K_test = selector.fit_transform(X_test, Y_test)

# Create Classifier objects
# sklean Decision tree uses CART
# Finding the reference for that is left as an exercise for the reader
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=10, min_samples_split=50)
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=10, min_samples_split=50)
clf_knn = KNeighborsClassifier(n_neighbors=20, n_jobs=4)
clf_adaB = RandomForestClassifier(min_samples_split=50)
clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate="adaptive", random_state=1, max_iter=500)
clf_SVC = SVC(kernel='linear', verbose=False, max_iter=-1, cache_size=7000)
clf_dict = {'GINI_DTree': clf_gini,
            'Entropy_Tree': clf_entropy,
            'k_NN': clf_knn,
            'AdaBoosting': clf_adaB,
            'Perceptron_ANN': clf_MLP,
            'SVC': clf_SVC}

# Testing for best parameters
for tester in range(1, 10):
    clf_SVC = SVC(C=tester, kernel='linear', verbose=False, max_iter=-1, cache_size=7000)
    cv = ShuffleSplit(n_splits=1, test_size=0.1, train_size=0.1, random_state=0)
    for train_index, test_index in cv.split(X, Y):
        for ti in train_index:
            S_train = X.iloc[ti, :]
        for ti2 in test_index:
            S_test = X.iloc[ti2, :]
        print(S_train.head(2))
    clf_SVC.fit(X_train, Y_train)
    print("SVC", tester, ": ", accuracy_score(Y_test, clf_SVC.predict(X_test)) * 100)

# Fit Classifiers to Data and Print results
print("\n--Adult Predictions--")

# Results Header
template = "{0:20}{1:15}{2:15}"
print(template.format("Classifier", "Accuracy(%)", "Runtime(s)"))
for key, value in clf_dict.items():
    if (value == clf_SVC):
        X_train = StandardScaler().fit_transform(X_train, Y_train)
    start = time.time()
    value.fit(X_train, Y_train)
    end = time.time()
    print(template.format(key, ("%.2f" % (accuracy_score(Y_test, value.predict(X_test)) * 100)), "%.2f" % (end - start)))

# # Scaling is terrible on this dataset. Disabling scaling moves
# # SVC from the worst performer to the best.
# scaler = StandardScaler()

# Plotter
for key, value in clf_dict.items():
    if (value == clf_SVC):
        cv = ShuffleSplit(n_splits=5, test_size=0.1, train_size=0.1 ,random_state=0)
        plt = plot_learning_curve(value, key, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=1)
    else:
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        plt = plot_learning_curve(value, key, X_train, Y_train, ylim=(0.0, 1.01), cv=cv, n_jobs=1)
    plt.savefig('Adult_' + key)

# # K-Series
# clf_gini.fit(K_train, Y_train)
# print("\nGini: ", accuracy_score(Y_test, clf_gini.predict(K_test)) * 100)
#
# clf_entropy.fit(K_train, Y_train)
# print("Entropy: ", accuracy_score(Y_test, clf_entropy.predict(K_test)) * 100)
#
# clf_adaB.fit(K_train, Y_train)
# print("AdaBoost: ", accuracy_score(Y_test, clf_adaB.predict(K_test)) * 100)
#
# clf_knn.fit(K_train, Y_train)
# print("kNN: ", accuracy_score(Y_test, clf_knn.predict(K_test)) * 100)
#
# clf_MLP.fit(K_train, Y_train)
# print("MLP: ", accuracy_score(Y_test, clf_MLP.predict(K_test)) * 100)
#
# clf_SVC.fit(K_train, Y_train)
# print("SVC: ", accuracy_score(Y_test, clf_SVC.predict(K_test)) * 100)



