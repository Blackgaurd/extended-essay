from edt.data.load import *
from edt.data.utils import train_test_split
from sklearn.tree import DecisionTreeClassifier

def run(dataset):
    (trainX, trainY), (testX, testY) = globals()[f"load_{dataset}"]().data()
    if testX is None or testY is None:
        trainX, trainY, testX, testY = train_test_split(trainX, trainY, 0.2)

    clf = DecisionTreeClassifier(max_depth=10)
    h = clf.fit(trainX, trainY)
