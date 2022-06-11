import json
import os
from pathlib import Path

from matplotlib import pyplot as plt

DIR = Path(os.path.dirname(os.path.abspath(__file__)))

with open(DIR / "history.json") as f:
    history = json.load(f)

x = [h["generation"] for h in history]

# average and max accuracy
y1 = [h["avg_accuracy"] * 100 for h in history]
y2 = [h["max_accuracy"] * 100 for h in history]
y3 = [h["avg_val_accuracy"] * 100 for h in history]
plt.plot(x, y1, label="Average")
plt.plot(x, y2, label="Max")
plt.plot(x, y3, label="Validation")
plt.xlabel("Generation")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend(loc="lower right")
plt.title("Accuracy")

plt.savefig(DIR / "accuracy.png")
plt.clf()

# average and max fitness
y1 = [h["avg_fitness"] for h in history]
y2 = [h["max_fitness"] for h in history]
plt.plot(x, y1, label="Average")
plt.plot(x, y2, label="Max")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.legend(loc="lower right")
plt.title("Fitness")

plt.savefig(DIR / "fitness.png")
plt.clf()
