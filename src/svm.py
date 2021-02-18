import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


cancer = datasets.load_breast_cancer()
classes = [['malignant', 'benign']]
x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)
y_prediction = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_prediction)

print("Accuracy: ", acc)
