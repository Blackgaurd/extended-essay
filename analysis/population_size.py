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


def run(seed, pop_size):
    random.seed(seed)
    clf = GDTClassifier(
        population_size=pop_size,
        split_probability=0.5,
        selectors=[(Tournament(4), 0.8), (Elitism(), 0.2)],
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
sizes = list(range(100, 501, 100))

product = list(itertools.product(range(3), sizes))
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
y = [mean(training_time[pop_size]) for pop_size in sizes]
ybars = [stdev(training_time[pop_size]) for pop_size in sizes]

ax = plt.subplot(gs[0, 0])
plt.title("Training time")
plt.ylabel("Time (s)")
plt.xlabel("Population size")
plt.errorbar(sizes, y, ybars, label="Training time")

# top right: test accuracy
y = [mean(testing_accuracy[pop_size]) * 100 for pop_size in sizes]
ybars = [stdev(testing_accuracy[pop_size]) * 100 for pop_size in sizes]

ax = plt.subplot(gs[0, 1])
plt.title("Test accuracy")
plt.ylabel("Accuracy (%)")
plt.xlabel("Population size")
plt.errorbar(sizes, y, ybars, label="Test accuracy")

# bottom span: accuracy convergence
ax = plt.subplot(gs[1, :])

x = list(avg_accuracy[sizes[0]].keys())
plt.title("Accuracy convergence")
plt.ylabel("Accuracy (%)")
plt.xlabel("Generation")
for sz, h in avg_accuracy.items():
    y = [mean(acc) * 100 for acc in h.values()]
    plt.plot(x, y, label=str(sz))
plt.legend(title="Population size")

plt.savefig("figs/population_size.png")
