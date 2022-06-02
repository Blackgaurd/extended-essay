from edt.data.load import *
from edt.data.utils import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

def run(dataset):
    data = globals()[f"load_{dataset}"]()
    (trainX, trainY), (testX, testY) = data.data()
    if testX is None or testY is None:
        trainX, trainY, testX, testY = train_test_split(trainX, trainY, 0.2)

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    clf = Sequential(
        [
            Dense(trainX.shape[1] * 2, activation="relu", input_shape=(trainX.shape[1],)),
            Dense(trainX.shape[1] * 4, activation="relu"),
            Dense(trainX.shape[1], activation="relu"),
            Dense(len(data.label_names), activation="softmax"),
        ]
    )

    clf.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h = clf.fit(trainX, trainY, epochs=10).history
