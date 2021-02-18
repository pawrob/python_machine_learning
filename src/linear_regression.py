import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from sklearn import linear_model
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
best_acc = 0
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# training model and selecting one with best accuracy
for _ in range(50):
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best_acc:
        best_acc = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: ", linear.coef_)
print("Int: ", linear.intercept_)

# predicting
predictions = linear.predict(x_test)
print("Acc: ", best_acc)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# plotting
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()
