""" This script is designed to train a classification algorithm to classify Titanic passengers into those who survived and those who didn't.

Author : William H. Knapp III
Date : 20 March 2016
Revised: 23 March 2016
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as sklm
from sklearn import metrics
from sklearn import cross_validation as cv
import csv as csv

#read in the data
df = pd.read_csv("train.csv")
dftest = pd.read_csv("test.csv")

#examine the data
#I'm using several functions as examples
df.describe()
df.columns
df.head()
df.tail(3)
df.info()

#Replace missing values in Age with the mean age
df.Age = df.Age.fillna(df.Age.mean())
dftest.Age = dftest.Age.fillna(df.Age.mean())

#Replace missing values in Embarked with the top location
df.Embarked = df.Embarked.fillna("S")

#Replace missing values in Fare with the median fare
dftest.Fare = dftest.Fare.fillna(dftest.Fare.median())

#Change the embarkation points to numeric values based on the
#the order of the ports.
#S: Southampton was first, so 0
#C: Cherbourg was second, so 1
#Q: Queenstown was last, so 2
df.Embarked[df.Embarked=='S'] = 0
df.Embarked[df.Embarked=='C'] = 1
df.Embarked[df.Embarked=='Q'] = 2
#A more concise way of doing this for the test set
dftest.Embarked = dftest.Embarked.map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Change sex to numeric
#male = 0
#female = 1
df.Sex[df.Sex=='male'] = 0
df.Sex[df.Sex=='female'] = 1
#A more concise way of doing this for the test set
dftest.Sex = pd.to_numeric(dftest.Sex.map( {'male': 0, 'female': 1} ))

#Now we need to convert the columns embarked and sex from objects to numeric
#We only need to do this for df, since we used mapping and conversion for the
#test dataframe
df.Sex = pd.to_numeric(df.Sex)
df.Embarked = pd.to_numeric(df.Embarked)


#Let's combine the number of relatives into a new column
df["Nrel"] = df.SibSp + df.Parch
dftest["Nrel"] = dftest.SibSp + dftest.Parch

#Let's create a variable to account for a curvelinear relationship
#between Age and Survived.
df["Age2"] = df.Age**2
dftest["Age2"] = dftest.Age**2

#Let's create a variable that takes the log of Fare as Fare is skew positive
#We'll add 1 to all the fares as there are fares that equal 0
df["LogFare"] = np.log10(df.Fare+1)
dftest["LogFare"] = np.log10(dftest.Fare+1)

#Let's create a variable that could account for an interaction between
#Age and Sex as there might be less of a bias to save older women.
#We'll add 1 to the Age variable so that the values for males aren't
#all zeros.
df["AgeSex"] = (df.Sex+1) * df.Age
dftest["AgeSex"] = (dftest.Sex+1) * dftest.Age

#Let's examine the correlation matrix
df.corr()

#Let's examine some individual correlations
np.corrcoef(df.Survived, df.Pclass) #-.338

#I'm breaking it down for ease of inspection
np.corrcoef(df.Survived, df.Sex) #.543

np.corrcoef(df.Survived, df.Age) #-.070
plt.scatter(df.Survived, df.Age)
plt.show()

np.corrcoef(df.Survived, df.SibSp) #-.035

np.corrcoef(df.Survived, df.Parch) #.082

np.corrcoef(df.Survived, df.Fare) #.257

np.corrcoef(df.Survived, df.Embarked) #.107

np.corrcoef(df.Survived, df.Nrel) #.017: Don't include in the model

np.corrcoef(df.Survived, df.Age2) #-.041: Don't include in the model

np.corrcoef(df.Survived, df.LogFare) #.330: This was a good one to create, it correlates more than Fare

np.corrcoef(df.Survived, df.AgeSex) #.307: This was also a good one to create

#Separate the features and the targets in the training set
features = df[["Pclass", "Sex", "Age", "SibSp",
               "Parch", "Fare", "Embarked",
               "LogFare", "AgeSex"]]
features_test = dftest[["Pclass", "Sex", "Age", "SibSp",
               "Parch", "Fare", "Embarked",
               "LogFare", "AgeSex"]]
target = df.Survived.values #Needed to convert from panda to 1-d numpy array for model fit

#Set our classification model
model = sklm.LogisticRegression(C=1e5)

#Fit our model on the training features and targets
np.random.seed(461)
model.fit(features,target)

#Get the predicted values for the training set
predicted = model.predict(features)

#Summarize the fit of the model
print(metrics.classification_report(target, predicted))
metrics.confusion_matrix(target, predicted)
#The average precision was .81 as was the recall and F1 score

#Let's cross validate using 5 folds.
scores = cv.cross_val_score(model, features, target, cv=5)

#Let's take the mean of scores to get the average accuracy
np.mean(scores) # .798: So the model seems to be behaving nicely

#Let's predict survival on our test set.
predicted_test = model.predict(features_test)

#Let's save our predictions for the test set
ids = dftest.PassengerId.values
f = open("titanic_test_predictions.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["PassengerId","Survived"])
writer.writerows(zip(ids, predicted_test))
f.close()