import numpy as np
import pandas as pd
from sklearn import metrics, neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

col_names= ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

data = pd.read_csv("car.data", names=col_names)

X = data[["buying", "maint", "safety"]].values
y = data[["class"]]

#convert the data

Le = LabelEncoder()

for i in range(X.shape[1]):
    X[:, i] = Le.fit_transform(X[:, i])



#convert the targets
label_mapping = {
    "unacc" : 0,
    "acc" : 1,
    "good" : 2,
    "vgood" : 3
}

y["class"] = y["class"].map(label_mapping)

y = np.array(y)

# create model

knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights="uniform")

# seperate for test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#train
knn.fit(X_train, y_train)

#predict
predictions = knn.predict(X_test)

# calculate accuracy
accuracy = metrics.accuracy_score(y_test, predictions)


print(f"Predictions : {predictions} with accuracy of \n {accuracy}")