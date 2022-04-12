import os

import pandas as pd

DIR = os.path.dirname(os.path.realpath(__file__))


def load_titanic():
    train = pd.read_csv(f"{DIR}/titanic/train.csv")
    test = pd.read_csv(f"{DIR}/titanic/test.csv")

    # drop unnecessary columns
    train_Y = train["Survived"]
    train = train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

    # fill missing data
    train["Age"].fillna(int(train["Age"].mean()), inplace=True)
    train["Embarked"].fillna("S", inplace=True)

    # replace non-numerical values with numbers
    train["Sex"].replace({"male": 0, "female": 1}, inplace=True)
    train["Embarked"].replace({"S": 0, "C": 1, "Q": 2}, inplace=True)

    # to numpy
    train_X = train.to_numpy().astype(float)
    train_Y = train_Y.to_numpy().astype(int)

    # repeat steps with test data
    test = test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]
    test["Age"].fillna(int(test["Age"].mean()), inplace=True)
    test["Embarked"].fillna("S", inplace=True)
    test["Sex"].replace({"male": 0, "female": 1}, inplace=True)
    test["Embarked"].replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
    test = test.to_numpy().astype(float)

    return (train_X, train_Y), (test, None)


def load_iris():
    train = pd.read_csv(f"{DIR}/iris/train.csv")

    train["Species"].replace(
        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}, inplace=True
    )

    trainX = (
        train[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
        .to_numpy()
        .astype(float)
    )
    trainY = train["Species"].to_numpy().astype(int)

    return (trainX, trainY), (None, None)
