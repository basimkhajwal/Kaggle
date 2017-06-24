import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

# Load data from CSV file
trainData = pd.read_csv("../data/titanic/train.csv")
trainLabels = trainData['Survived'].as_matrix()

# Remove irrelevant columns and labels from the data, and convert to numerical
trainData.drop(['Name', 'Survived', 'Cabin', 'Embarked', 'Ticket'], axis=1, inplace=True)
trainData["Age"] = trainData["Age"].map(lambda a: 30 if np.isnan(a) else a)
trainData["Sex"] = trainData["Sex"].map({"male": 0, "female": 1})

trainData = trainData.as_matrix()

logModel = LogisticRegression().fit(trainData, trainLabels)
print(trainData.shape[0] * logModel.score(trainData, trainLabels))
