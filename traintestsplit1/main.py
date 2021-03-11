from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


iris = datasets.load_iris()
#split in to features and labels
X = iris.data
y = iris.target

# split train and test with 20% ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 