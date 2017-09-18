import time
import os
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
from sklearn.model_selection import train_test_split
from LearningCurvePlot import plot_learning_curve

# Read Training set
trainFile = open(os.environ['USERPROFILE'] + '//OneDrive//Documents//Education//Grad School//Datasets//adult_edit.csv')
train = pd.read_csv(trainFile)
train = train.fillna('')
train.index.name = 'Row'
print("Train dataset shape: ", train.shape)

# Read Test set
testFile = open(os.environ['USERPROFILE'] + '//OneDrive//Documents//Education//Grad School//Datasets//adultCV.csv')
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

# Create X/Y sets
X = df_with_dumbo.drop('income_bin', axis=1)
Y = df_with_dumbo['income_bin']

# Create Test/Train sets
X_train = X.loc['train']
Y_train = Y.loc['train']
X_test = X.loc['test']
Y_test = Y.loc['test']

# Feature Selection Comparison
# selector = SelectKBest(chi2, k=20)
# X_train = selector.fit_transform(X_train, Y_train)
# X_test = selector.fit_transform(X_test, Y_test)

# Create Classifier objects
# sklean Decision tree uses CART
# Finding the reference for that is left as an exercise for the reader
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=10, min_samples_split=50)
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=10, min_samples_split=50)
clf_knn = KNeighborsClassifier(n_neighbors=20, n_jobs=4)
clf_adaB = RandomForestClassifier(min_samples_split=50)
clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate="adaptive", random_state=1, max_iter=500)
clf_SVC = SVC(C=60, kernel='linear', verbose=False, max_iter=-1, cache_size=7000)
clf_dict = {'GINI_DTree': clf_gini,
            'Entropy_Tree': clf_entropy,
            'k_NN': clf_knn,
            'AdaBoosting': clf_adaB,
            'Perceptron_ANN': clf_MLP,
            'SVC': clf_SVC}

# Testing for best parameters
# S_train, S_test = train_test_split(df_with_dumbo, test_size=0.97)
# SX_train = S_train.drop('income_bin', axis=1)
# SY_train = S_train['income_bin']
# SX_test = S_test.drop('income_bin', axis=1)
# SY_test = S_test['income_bin']

# Feature Selection
# selector = SelectKBest(chi2, k=4)
# TX_train = selector.fit_transform(X_train, Y_train)
# TX_test = selector.fit_transform(X_test, Y_test)

# Testing
# for tester in range(2, 10):
#     TX_train = StandardScaler().fit_transform(TX_train, SY_train)
#     print("Training on", TX_train.shape, "samples")
#
#     clf_SVC = SVC(C=60, kernel='linear', verbose=False, max_iter=-1, cache_size=7000)
#     clf_SVC.fit(TX_train, SY_train)
#     print("SVC", tester, ": ", accuracy_score(SY_test, clf_SVC.predict(TX_test)) * 100)

# Fit Classifiers to Data and Print results
print("\n--Adult Predictions--")

# Results Header
template = "{0:20}{1:15}{2:15}"
print(template.format("Classifier", "Accuracy(%)", "Runtime(s)"))
for key, value in clf_dict.items():
    # SVC is special
    if (value == clf_SVC):
        # Form smaller training set for SVC
        S_train, S_test = train_test_split(df_with_dumbo, test_size=0.98)

        # Split into X/Y sets
        SX_train = S_train.drop('income_bin', axis=1)
        SY_train = S_train['income_bin']
        SX_test = S_test.drop('income_bin', axis=1)
        SY_test = S_test['income_bin']

        # Choose k-best features from the SX sets
        selector = SelectKBest(chi2, k=6)
        TX_train = selector.fit_transform(SX_train, SY_train)
        TX_test = selector.fit_transform(SX_test, SY_test)

        # Scale, if it helps
        TX_train = StandardScaler().fit_transform(TX_train, SY_train)

        start = time.time()
        value.fit(TX_train, SY_train)
        end = time.time()
        print(template.format(key, ("%.2f" % (accuracy_score(SY_test, value.predict(TX_test)) * 100)), "%.2f" % (end - start)))
    else:
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
        cv = ShuffleSplit(n_splits=5, test_size=0.98, random_state=0)
        plt = plot_learning_curve(value, key, X, Y, ylim=(0.0, 1.01), cv=cv, n_jobs=1)
    else:
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        plt = plot_learning_curve(value, key, X, Y, ylim=(0.0, 1.01), cv=cv, n_jobs=1)
    print(key, "figure complete")
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



