from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
import pandas as pd
import numpy as np

#load data
bc = load_breast_cancer()

#assign data and labels
X = np.array(preprocessing.scale(bc["data"]))  #scale data 
y = np.array(bc["target"])
labels = bc["target_names"]

#split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#create model
model = KMeans(n_clusters=2, random_state=0)
model.fit(X_train)


# test the data

predictions = model.predict(X_test) 
accuracy = metrics.accuracy_score(y_test, predictions)

print(y)
print(f"Predicitons : {predictions}\n with a accuracy of {accuracy}")