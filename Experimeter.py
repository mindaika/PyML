import os
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from HelperFunctions import plot_learning_curve


def parameter_search():
    template = "{0:25}{1:15}{2:15}{3:15}{4:15}"
    print(template.format("\nClassifier", "Accuracy(%)", "Runtime(s)", "i * scalar", "Predict(s)"))

    for i in range(1, 10):
        scalar = 10

        test_value = (i * scalar)
        # float(str(i / 10))
        X_train, X_test, Y_train, Y_test = train_test_split(X_yeast, Y_yeast, train_size=0.65, random_state=0)

        _estimator = SVC(kernel='linear')
        _estimator.C = (i * scalar)

        start = time.time()
        _estimator.fit(X_train, Y_train)
        end = time.time()

        start_predict = time.time()
        prediction = (accuracy_score(Y_test, _estimator.predict(X_test)))
        end_predict = time.time()

        print(template.format('Test value=' + str(test_value),
                              "%.2f" % (prediction * 100),
                              "%.2f" % (end - start),
                              str(i * scalar),
                              "%.2f" % (end_predict - start_predict)))


# -- File Loaders --
# Read Adult Training set
adult_trainFile = open(
    os.environ['USERPROFILE'] + '//OneDrive//Documents//Education//Grad School//Datasets//adult_edit.csv')
adult_train = pd.read_csv(adult_trainFile)
adult_train = adult_train.fillna('')
adult_train.index.name = 'Row'
print("Adult train dataset shape: ", adult_train.shape)

# Read Adult Test set
adult_testFile = open(
    os.environ['USERPROFILE'] + '//OneDrive//Documents//Education//Grad School//Datasets//adultCV.csv')
adult_test = pd.read_csv(adult_testFile)
adult_test = adult_test.fillna('')
adult_test.index.name = 'Row'
print("Adult test dataset shape: ", adult_test.shape)

# Read Yeast
yeast_trainFile = open(os.environ['USERPROFILE'] + '//OneDrive//Documents//Education//Grad School//Datasets//yeast.csv')
yeast = pd.read_csv(yeast_trainFile)
print("Yeast dataset shape: ", yeast.columns.values)

# -- Adult Pre-processing --
# Concatenate Adult sets
df_adult = pd.concat([adult_train, adult_test], keys=["train", "test"])

# Booleate (real word) sex, income_bin
# Sex: Female=0, Male=1
# Income_bin: <=50K=0, >50=1
df_adult['income_bin'] = df_adult['income_bin'].apply(lambda income: 0 if income == '<=50K' else 1)
df_adult['sex'] = df_adult['sex'].apply(lambda sex: 0 if sex == "Male" else 1)

# Encode dataset features
labels_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
adult = pd.get_dummies(df_adult, labels_to_encode)

# -- Yeast Pre-processing --
# Encoding
# LabelEncoding should be fine for the DV
enc = LabelEncoder()
yeast.loc[:, 'SequenceName'] = enc.fit_transform(yeast.loc[:, 'SequenceName'])
yeast.set_index(['SequenceName'], inplace=True)
yeast.loc[:, 'ClassDist'] = enc.fit_transform(yeast.loc[:, 'ClassDist'])

# -- DV/IV Splitting --
X_adult = adult.drop('income_bin', axis=1)
Y_adult = adult.loc[:, 'income_bin']

X_yeast = yeast.drop('ClassDist', axis=1)
Y_yeast = yeast.loc[:, 'ClassDist']

# -- Feature Selection --
# Feature Selection Comparison
# selector = SelectKBest(chi2, k=20)
# X_train = selector.fit_transform(X_train, Y_train)
# X_test = selector.fit_transform(X_test, Y_test)


# -- Classifier setup --
adult_gini = DecisionTreeClassifier(criterion="gini",
                                    random_state=100,
                                    max_depth=60,
                                    min_samples_leaf=50,
                                    min_samples_split=500)
adult_entropy = DecisionTreeClassifier(criterion="entropy",
                                       random_state=100,
                                       max_depth=10,
                                       min_samples_leaf=50,
                                       min_samples_split=500)
adult_knn = KNeighborsClassifier(n_neighbors=30,
                                 n_jobs=4)
adult_adaB = RandomForestClassifier(max_depth=20,
                                    min_samples_split=50,
                                    min_samples_leaf=4,
                                    max_leaf_nodes=500)
adult_MLP = MLPClassifier(hidden_layer_sizes=(3, 2, 4),
                          solver='sgd',
                          max_iter=500)
adult_SVC = SVC(C=1.0,
                kernel='rbf',
                verbose=False,
                max_iter=-1,
                cache_size=1000)
yeast_gini = DecisionTreeClassifier(criterion="gini",
                                    max_depth=9,
                                    min_samples_leaf=10,
                                    min_samples_split=50)
yeast_entropy = DecisionTreeClassifier(criterion="entropy",
                                       max_depth=10,
                                       min_samples_split=30,
                                       min_samples_leaf=30)
yeast_knn = KNeighborsClassifier(n_neighbors=18,
                                 n_jobs=4)
yeast_RFC = RandomForestClassifier(max_depth=8,
                                   min_samples_split=50,
                                   min_samples_leaf=10,
                                   max_leaf_nodes=100)
yeast_MLP = MLPClassifier(max_iter=4000)
yeast_SVC = SVC(C=10,
                kernel='linear',
                verbose=False,
                max_iter=-1,
                cache_size=1000)

clf_dict = {'Adult_gini DTree': adult_gini,
            'Adult_entropy_Tree': adult_entropy,
            'Adult_k-NN': adult_knn,
            'Adult_RandomForest': adult_adaB,
            'Adult_Perceptron_ANN': adult_MLP,
            'Adult_SVC': adult_SVC,
            'Yeast_gini_DTree': yeast_gini,
            'Yeast_entropy_Tree': yeast_entropy,
            'Yeast_k-NN': yeast_knn,
            'Yeast_RandomForest': yeast_RFC,
            'Yeast_Perceptron_ANN': yeast_MLP,
            'Yeast_SVC': yeast_SVC}

# -- Parameter Testser --
# parameter_search()


# -- Classifier fitting --
# Results Header
template = "{0:25}{1:15}{2:15}{3:15}"
print(template.format("\nClassifier", "Accuracy (%)", "Learning Time (s)", "Predict Time (s)"))
for key, value in clf_dict.items():
    if key[0:5] == 'Adult':
        X = X_adult
        Y = Y_adult
    else:
        X = X_yeast
        Y = Y_yeast

    if key[-3:] == 'SVC':
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=200)
        X_train = scale(X_train)
        X_test = scale(X_test)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.45)

    start = time.time()
    value.fit(X_train, Y_train)
    end = time.time()

    start_predict = time.time()
    prediction = (accuracy_score(Y_test, value.predict(X_test)))
    end_predict = time.time()

    print(template.format(key,
                          "%.2f" % (prediction * 100),
                          "%.2f" % (end - start),
                          "%.2f" % (end_predict - start_predict)))

# -- Results Plotting --
for key, value in clf_dict.items():
    if key[0:5] == 'Adult':
        X = X_adult
        Y = Y_adult
    else:
        X = X_yeast
        Y = Y_yeast

    if key[-3:] == 'SVC':
        X = scale(X)
        cv = ShuffleSplit(n_splits=20, train_size=200)
    else:
        cv = ShuffleSplit(n_splits=100, test_size=0.45)

    start_plot = time.time()
    plt = plot_learning_curve(value, key, X, Y, ylim=(0.0, 1.01), cv=cv, n_jobs=1)
    end_plot = time.time()

    plt.savefig(key)
    print(key, "figure complete. Plot time: " + "%.2f" % (end_plot - start_plot))
