from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


boston = datasets.load_boston()

# features / labels

X = boston["data"]
y = boston["target"]


# create model
l_reg = linear_model.LinearRegression()

# visualize some feature vs target

fet = 5 # feature 
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(X[:, fet], y, "x")

ax.set_ylabel("Median value of owner-occupied homes in $1000's")
ax.set_xlabel(boston["feature_names"][fet])

# plt.show()

#split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train - test

model = l_reg.fit(X_train, y_train)
predictions = l_reg.predict(X_test)
# R values
print(f"Predictions : {predictions}, R value : {l_reg.score(X, y)}")  
#print parameters
print(f"Coefficients : {l_reg.coef_}")
# print intercepts ???
print(f"Intercepts : {l_reg.intercept_}")

