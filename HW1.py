from sklearn import tree
import pandas as pd
import numpy as np
import graphviz


# Read Training set
trainFile = open('C://Users//rs//OneDrive//Documents//Education//Grad School//Datasets//adult_edit.csv')
train = pd.read_csv(trainFile)
train = train.fillna('')
train.index.name = 'Row'

# Read Test set
testFile = open('C://Users//rs//OneDrive//Documents//Education//Grad School//Datasets//adultCV.csv')
test = pd.read_csv(testFile)
test = test.fillna('')
test.index.name = 'Row'

# Munging test area
train_obj = train.select_dtypes(include=['object'])
train[train_obj.columns] = train_obj.apply(lambda x: x.str.lstrip())
train = train.replace(['White'], ['Honkey'])
# print(train['sex'][2] == 'Male')
# train['sex'] = train['sex'].replace(['Male', 'Female'], ['penis-haver', 'vagine-haver'])
print(train.head(4))


# Slice data
X = train.values[:, 0:13]
Y = train.values[:, 14]

# Create DecisionTreeClassifier
# sklean Decision tree uses CART
# Finding the reference for that is left as an exercise for the reader
J48 = tree.DecisionTreeClassifier()
# J48 = J48.fit(train, test['income_bin'])

# dot_data = tree.export_graphviz(J48, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("adult")
