import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("../datasets/car.data")

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"
names = ["unacc", "acc", "good", "vgood"]
accuracy = 0
X = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)
model = KNeighborsClassifier(n_neighbors=7)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predicted = model.predict(x_test)

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "\tData: ", x_test[x], "Actual: ", names[y_test[x]])
    if predicted[x] == y_test[x]:
        accuracy += 1

accuracy = accuracy / len(predicted)

print("Accuracy based on sklearn: ", acc)
print("Accuracy based on result:  ", accuracy)
