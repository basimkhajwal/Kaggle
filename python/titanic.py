import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import *
import time

# Load data from CSV file
trainData = pd.read_csv("../data/titanic/train.csv")
trainLabels = trainData['Survived'].as_matrix()

# Remove irrelevant columns and labels from the data, and convert to numerical
trainData.drop(['Name', 'Survived', 'Cabin', 'Embarked', 'Ticket'], axis=1, inplace=True)
trainData["Age"] = trainData["Age"].map(lambda a: 30 if np.isnan(a) else a)
trainData["Sex"] = trainData["Sex"].map({"male": 0, "female": 1})
trainData = trainData.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(trainData, trainLabels, test_size=0.2)

def scoreModel(name, model):
    before = time.time()
    model.fit(X_train, y_train)
    timeTaken = time.time() - before
    score = X_test.shape[0] * model.score(X_test, y_test)

    print(name," -- ", timeTaken, " -- ", score)

# score models for use in ensemble learning
scoreModel("Logistic Regression", LogisticRegressionCV())
scoreModel("Random Forest", RandomForestClassifier()) # performs best for this data set
scoreModel("Adaptive Boosting", AdaBoostClassifier())
scoreModel("SVM", SVC()) # not linearly separable, so SVM doesnt perform well
