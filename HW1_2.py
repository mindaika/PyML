import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Read Training set
trainFile = open('C://Users//Randall//OneDrive//Documents//Education//Grad School//Datasets//adult_edit.csv')
train = pd.read_csv(trainFile)
train = train.fillna('')
train.index.name = 'Row'
print("Training dataset shape: ", train.shape)

# Read Test set
testFile = open('C://Users//Randall//OneDrive//Documents//Education//Grad School//Datasets//adultCV.csv')
test = pd.read_csv(testFile)
test = test.fillna('')
test.index.name = 'Row'
print("Training dataset shape: ", test.shape)

# Concatenate sets for pre-processing
df = pd.concat([train, test], keys=["train", "test"])

# Debugging
# print(df.columns.values)
# for val in df.columns.values:
#     print("\n", val, df[val].head(3))

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
clf_SVC = SVC(kernel='linear', verbose=True, max_iter=-1)

# for tester in range(100, 101):
#     clf_MLP = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(9, 5), learning_rate="adaptive", random_state=1, max_iter=500)
#     clf_MLP.fit(X_train, Y_train)
#     print("MLP", tester, ": ", accuracy_score(Y_test, clf_MLP.predict(X_test)) * 100)

# Fit Classifiers to Data and Print results
# X-Series
print("\n--X-Series--")

clf_gini.fit(X_train, Y_train)
print("Gini: ", accuracy_score(Y_test, clf_gini.predict(X_test)) * 100)

clf_entropy.fit(X_train, Y_train)
print("Entropy: ", accuracy_score(Y_test, clf_entropy.predict(X_test)) * 100)

clf_adaB.fit(X_train, Y_train)
print("AdaBoost: ", accuracy_score(Y_test, clf_adaB.predict(X_test)) * 100)

clf_knn.fit(X_train, Y_train)
print("kNN: ", accuracy_score(Y_test, clf_knn.predict(X_test)) * 100)

clf_MLP.fit(X_train, Y_train)
print("MLP: ", accuracy_score(Y_test, clf_MLP.predict(X_test)) * 100)

scaler = StandardScaler()
clf_SVC.fit(scaler.fit_transform(K_train), Y_train)
print("SVC: ", accuracy_score(Y_test, clf_SVC.predict(K_test)) * 100)


# K-Series
clf_gini.fit(K_train, Y_train)
print("\nGini: ", accuracy_score(Y_test, clf_gini.predict(K_test)) * 100)

clf_entropy.fit(K_train, Y_train)
print("Entropy: ", accuracy_score(Y_test, clf_entropy.predict(K_test)) * 100)

clf_adaB.fit(K_train, Y_train)
print("AdaBoost: ", accuracy_score(Y_test, clf_adaB.predict(K_test)) * 100)

clf_knn.fit(K_train, Y_train)
print("kNN: ", accuracy_score(Y_test, clf_knn.predict(K_test)) * 100)

clf_MLP.fit(K_train, Y_train)
print("MLP: ", accuracy_score(Y_test, clf_MLP.predict(K_test)) * 100)

clf_SVC.fit(K_train, Y_train)
print("SVC: ", accuracy_score(Y_test, clf_SVC.predict(K_test)) * 100)


# title = "Learning Curves (Decision Tree)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)

# plt = plot_learning_curve(clf_gini, title, K_train, Y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

# title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
# plt = plot_learning_curve(estimator, title, X, Y, (0.7, 1.01), cv=cv, n_jobs=1)

# plt.show()



