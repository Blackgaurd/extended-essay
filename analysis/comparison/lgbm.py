import random
import time

import lightgbm as lgbm
import numpy as np
from gdt.data.load import load_mushroom
from gdt.data.utils import train_test_split

random.seed(123)
np.random.seed(123)

(trainX, trainY), (testX, testY) = load_mushroom().data()
trainX, trainY, testX, testY = train_test_split(trainX, trainY, 0.4)
testX, testY, valX, valY = train_test_split(testX, testY, 0.5)

clf = lgbm.LGBMClassifier(max_depth=5, n_estimators=400, random_state=123)
start = time.time()
clf.fit(trainX, trainY)
end = time.time()

res = clf.predict(testX)
accuracy = np.mean(res == testY)
print(f"Accuracy: {accuracy}")
print(f"Time: {end - start}")
