import itertools
from multiprocessing import Pool
import random
import time
from collections import defaultdict
from statistics import mean, stdev

from gdt.classifier import GDTClassifier
from gdt.data.load import load_mushroom
from gdt.data.utils import train_test_split
from gdt.fitness import FitnessEvaluator, negative_quadratic
from gdt.selection import Elitism, Tournament
from matplotlib import gridspec
from matplotlib import pyplot as plt

data = load_mushroom()
(X, y), (X_test, y_test) = data.data()
X, y, X_test, y_test = train_test_split(X, y, 0.25)


def run(seed, tournament_p):
    random.seed(seed)
    clf = GDTClassifier(
        population_size=200,
        split_probability=0.5,
        selectors=[(Tournament(4), tournament_p), (Elitism(), 1 - tournament_p)],
        mutation_probability=0.1,
        optimal_depth=4,
        fitness_evaluator=FitnessEvaluator(0.9, 0.1, negative_quadratic),
    )

    start = time.time()
    h = clf.fit(X, y, generations=100, get_stats=5)
    end = time.time()

    predictions = clf.predict(X_test)

    return {
        "history": h,
        "training_time": end - start,
        "test_accuracy": sum(
            predictions[i] == y_test[i] for i in range(len(predictions))
        )
        / len(y_test),
    }


training_time = defaultdict(list)
testing_accuracy = defaultdict(list)
avg_accuracy = defaultdict(lambda: defaultdict(list))
tps = [0, 0.25, 0.5, 0.75, 1.0]

product = list(itertools.product(range(3), tps))
with Pool() as executor:
    for (seed, c1), res in zip(product, executor.starmap(run, product)):
        training_time[c1].append(res["training_time"])
        testing_accuracy[c1].append(res["test_accuracy"])
        for h in res["history"]:
            avg_accuracy[c1][h["generation"]].append(h["avg_accuracy"])

gs = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=plt.subplot(111), hspace=0.5, wspace=0.5
)
plt.figure()

# top left: training time
y = [mean(training_time[tournament_p]) for tournament_p in tps]
ybars = [stdev(training_time[tournament_p]) for tournament_p in tps]

ax = plt.subplot(gs[0, 0])
plt.title("Training time")
plt.ylabel("Time (s)")
plt.xlabel("Tournament selection (%)")
plt.errorbar(tps, y, ybars, label="Training time")

# top right: test accuracy
y = [mean(testing_accuracy[tournament_p]) * 100 for tournament_p in tps]
ybars = [stdev(testing_accuracy[tournament_p]) * 100 for tournament_p in tps]

ax = plt.subplot(gs[0, 1])
plt.title("Test accuracy")
plt.ylabel("Accuracy (%)")
plt.xlabel("Tournament selection (%)")
plt.errorbar(tps, y, ybars, label="Test accuracy")

# bottom span: accuracy convergence
ax = plt.subplot(gs[1, :])

x = list(avg_accuracy[tps[0]].keys())
plt.title("Accuracy convergence")
plt.ylabel("Accuracy (%)")
plt.xlabel("Generation")
for sz, h in avg_accuracy.items():
    y = [mean(acc) * 100 for acc in h.values()]
    plt.plot(x, y, label=str(sz))
plt.legend(title="Tournament selection (%)")

plt.savefig("figs/selection_distribution.png")
