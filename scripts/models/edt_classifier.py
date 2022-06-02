from edt.classifier import EDTClassifier
from edt.data.load import *
from edt.data.utils import train_test_split
from edt.fitness import FitnessEvaluator, negative_quadratic
from edt.selection import Elitism, Tournament


def run(dataset):
    (trainX, trainY), (testX, testY) = globals()[f"load_{dataset}"]().data()
    if testX is None or testY is None:
        trainX, trainY, testX, testY = train_test_split(trainX, trainY, 0.2)

    kwargs = {
        "population_size": 100,
        "split_probability": 0.5,
        "selectors": [(Tournament(4), 0.8), (Elitism(), 0.2)],
        "crossover_probability": 0.8,
        "mutation_probability": 0.1,
        "optimal_depth": 5,
        "fitness_evaluator": FitnessEvaluator(0.9, 0.1, negative_quadratic),
    }

    clf = EDTClassifier(**kwargs)
    h = clf.fit(trainX, trainY, 50, verbose=10)
