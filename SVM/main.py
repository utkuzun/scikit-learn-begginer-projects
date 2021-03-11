from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn import metrics

iris = datasets.load_iris()
#split in to features and labels
X = iris.data
y = iris.target

#classes array

classes = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

# split train and test with 20% ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# init model and train

model = svm.SVC()
model.fit(X_train, y_train)


print(model)

# predict and evaluate
predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(predictions, y_test)

print(f"Predictions: {predictions}\n with accuracy : {accuracy}")