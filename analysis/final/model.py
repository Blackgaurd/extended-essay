import json
import os
import pickle as pkl
import random
from pathlib import Path

import numpy as np
from gdt.classifier import GDTClassifier
from gdt.data.load import load_mushroom
from gdt.data.utils import train_test_split
from gdt.fitness import FitnessEvaluator, negative_quadratic
from gdt.selection import Elitism, Tournament

DIR = Path(os.path.dirname(os.path.abspath(__file__)))
random.seed(123)
np.random.seed(123)

data = load_mushroom()
(trainX, trainY), (testX, testY) = data.data()
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

history = []
for i in range(4):
    h = clf.fit(
        trainX,
        trainY,
        generations=25,
        valX=valX,
        valY=valY,
        feature_names=data.feature_names,
        label_names=data.label_names,
        get_stats=1,
    )
    for j in range(len(h)):
        h[j]["generation"] += i * 25
    history.extend(h)
    # population is sorted with best fitness first
    clf.population[0].visualize(str(DIR / f"generation_{(i+1)*25}.png"))
    print(f"Generation {(i+1)*25}/100")

with open(DIR / "history.json", "w") as f:
    json.dump(history, f, indent=4)

with open(DIR / "clf.pkl", "wb") as f:
    pkl.dump(clf, f)
