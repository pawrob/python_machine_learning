import sklearn
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("../datasets/car.data")
print(data.head)

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying,maint,doors,persons,lug_boot,safety,cls))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
