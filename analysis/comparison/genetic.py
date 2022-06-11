import random
import time

import numpy as np
from gdt.classifier import GDTClassifier
from gdt.data.load import load_mushroom
from gdt.data.utils import train_test_split
from gdt.fitness import FitnessEvaluator, negative_quadratic
from gdt.selection import Elitism, Tournament

random.seed(123)
np.random.seed(123)

(trainX, trainY), (testX, testY) = load_mushroom().data()
trainX, trainY, testX, testY = train_test_split(trainX, trainY, 0.4)
testX, testY, valX, valY = train_test_split(testX, testY, 0.5)

clf = GDTClassifier(
    population_size=400,
    split_probability=0.5,
    selectors=[(Tournament(10), 0.8), (Elitism(), 0.2)],
    mutation_probability=0.2,
    optimal_depth=5,
    fitness_evaluator=FitnessEvaluator(0.7, 0.3, negative_quadratic),
)
start = time.time()
clf.fit(trainX, trainY, generations=100)
end = time.time()

res = np.array(clf.predict(testX))
accuracy = np.mean(res == testY)
print(f"Accuracy: {accuracy}")
print(f"Time: {end - start}")
