import time
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
import time
import os
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



# Read Adult Training set
adult_trainFile = open(os.environ['USERPROFILE'] + '//OneDrive//Documents//Education//Grad School//Datasets//adult_edit.csv')
adult_train = pd.read_csv(adult_trainFile)
adult_train = adult_train.fillna('')
adult_train.index.name = 'Row'
print("Adult train dataset shape: ", adult_train.shape)

# Read Adult Test set
adult_testFile = open(os.environ['USERPROFILE'] + '//OneDrive//Documents//Education//Grad School//Datasets//adultCV.csv')
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
