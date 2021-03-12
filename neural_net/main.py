from PIL import Image
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import scipy.io
from sklearn.model_selection import train_test_split

#create training variables

data1 = scipy.io.loadmat('NN_ex4/ex4data1.mat')
data2 = scipy.io.loadmat('NN_ex4/ex4weights.mat')
X = np.array(data1["X"])          # 5000 samples with 400 parameters(20x20 gray scale) 
y = data1["y"]          # target as 0-9 digits
y[y==10] = 0            # convert 10s to 0s


# create test and train variables 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

#create model

clf = MLPClassifier(solver = "lbfgs", activation="relu", hidden_layer_sizes=(20, 20))

#train model

clf.fit(X_train, y_train)

#accuracy 

predicts = clf.predict(X_test)
acc = confusion_matrix(y_test, predicts)

def accuracy(cm):
    diagonal = cm.trace()
    elements = np.sum(cm)
    return diagonal/elements

print(f"Accuracy : {accuracy(acc)}")
