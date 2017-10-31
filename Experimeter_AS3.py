import csv
import os

import numpy as np
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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn import metrics
from time import time




from HelperFunctions import plot_learning_curve


def parameter_search():
    template = "{0:25}{1:15}{2:15}{3:15}{4:15}"
    print(template.format("\nClassifier", "Accuracy(%)", "Runtime(s)", "i * scalar", "Predict(s)"))

    for i in range(1, 10):
        scalar = 10

        test_value = (i * scalar)
        # float(str(i / 10))
        X_train, X_test, Y_train, Y_test = train_test_split(X_yeast, Y_yeast, train_size=0.65)

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
print("Yeast dataset shape: ", yeast.shape)

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

# -- Classifier setup --
default_yeast_KMeans = KMeans(n_clusters=10, random_state=13)
default_adult_KMeans = KMeans(n_clusters=2, random_state=13)
default_adult_GMM = GaussianMixture(random_state=13)
default_yeast_GMM = GaussianMixture(random_state=13)

clf_dict = {'Yeast Default GMM': default_yeast_GMM,
            'Adult Default GMM': default_adult_GMM,
            'Yeast Default KMeans': default_yeast_KMeans,
            'Adult Default KMeans': default_adult_KMeans
            }

# -- Parameter Testser --
# parameter_search()

# -- Testing --
n_samples, n_features = X_yeast.shape
n_labels = len(np.unique(Y_yeast))
labels = Y_yeast

sample_size = 300

print("n_labels: %d, \t n_samples %d, \t n_features %d"
      % (n_labels, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_labels, n_init=10),
              name="k-means++", data=X_yeast)

bench_k_means(KMeans(init='random', n_clusters=n_labels, n_init=10),
              name="random", data=X_yeast)


# Results Header
template = "{0:35}{1:15}{2:15}{3:15}"
print(template.format("\nClassifier", "Accuracy (%)", "Learning Time (s)", "Predict Time (s)"))

output_file = open('.//results//' + 'hw1_results.csv', 'w', newline='')
csv_writer = csv.writer(output_file, delimiter=',')
csv_writer.writerow(["Classifier", "Accuracy (%)", "Learning Time (s)", "Predict Time (s)"])

for key, value in clf_dict.items():
    if key[0:5] == 'Adult':
        X = X_adult
        Y = Y_adult
    else:
        X = X_yeast
        Y = Y_yeast

    X = scale(X)

    # -- Feature Selection --
    # Feature Selection Comparison
    # selector = SelectKBest(chi2, k=100)
    # X_train = selector.fit_transform(X_train, Y_train)
    # X_test = selector.fit_transform(X_test, Y_test)

    start = time.time()
    fitted = value.fit(X)
    end = time.time()

    start_predict = time.time()
    prediction = value.predict(X)
    end_predict = time.time()

    # Scoring
    correct = 0
    for i in Y:
        if Y[i] == prediction[i]:
            correct += 1

    # Display Results
    print(template.format(key,
                          "%.2f" % (correct / Y.shape[0] * 100),
                          "%.2f" % (end - start),
                          "%.2f" % (end_predict - start_predict)))

    # Write results file
    csv_writer.writerow([key, prediction, (end - start), (end_predict - start_predict)])

    # -- Results Plotting --
    start_plot = time.time()

    # plt, ax = plt.subplots(figsize=(9, 7))
    # ax.scatter(X[:, 0], X[:, 1], c=value.predict(X_test), s=50, cmap='viridis')
    #
    # # get centers for plot
    # centers = value.cluster_centers_
    # ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75)
    # plt.title('sklearn k-means', fontsize=18, fontweight='demi')

    end_plot = time.time()

    # savefile = './/results//' + key
    # plt.savefig(savefile)
    # print(key, "figure complete. Plot time: " + "%.2f" % (end_plot - start_plot))
    # plt.close()
output_file.close()
