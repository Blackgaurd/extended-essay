from edt.data.load import *
from edt.data.utils import train_test_split
from sklearn.ensemble import RandomForestClassifier

def run(dataset):
    (trainX, trainY), (testX, testY) = globals()[f"load_{dataset}"]().data()
    if testX is None or testY is None:
        trainX, trainY, testX, testY = train_test_split(trainX, trainY, 0.2)

    clf = RandomForestClassifier(max_depth=6)
    clf.fit(trainX, trainY)
