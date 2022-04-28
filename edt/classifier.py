from collections import deque
import copy
import random
import statistics
from typing import List

import numpy as np

data_loaded = False

random.seed(2)


def load_features(trainX: np.ndarray, trainY: np.ndarray):
    global features, feature_vals, labels, data_loaded

    # features and feature values
    features = list(range(trainX.shape[1]))  #! watch for this
    feature_vals = [list(set(trainX[:, i])) for i in features]

    # unique labels
    labels = list(set(trainY))

    data_loaded = True


class Node:
    def __init__(self, split_val: float = None, feature: int = None, label: int = None):
        self.split_val = split_val
        self.feature = feature
        self.label = label

    @classmethod
    def leaf(cls, label: int):
        return cls(label=label)

    @classmethod
    def split(cls, feature: int, split_val: float):
        return cls(feature=feature, split_val=split_val)

    def is_leaf(self):
        return self.label is not None


class DecisionTree:
    def __init__(self, max_depth: int):
        # store nodes in list for fast random access
        # lists are 1 indexed

        self.max_depth = max_depth
        l = 2 ** (max_depth + 3)  # extra space for crossover

        self.nodes = [None for _ in range(l)]
        self.depth_l = [0 for _ in range(l)]  # TODO: store node depth in node class

        # root has depth 0
        root_feature = random.choice(features)
        self.nodes[1] = Node.split(
            root_feature, random.choice(feature_vals[root_feature])
        )
        self.depth = 0

    @classmethod
    def generate_random(cls, max_depth: int, split_p: float):
        assert 0 <= split_p < 1
        ret = cls(max_depth)

        # generate tree with bfs
        q = deque([1])
        while q:
            cur = q.popleft()

            # loop twice for left and right child
            for _ in range(2):
                if cur * 4 < 2 ** (max_depth + 1) and split_p <= random.random():
                    next_pos = ret.add_node(cur, "split")
                    q.append(next_pos)
                else:
                    ret.add_node(cur, "leaf")

        return ret

    def add_node(self, node_ind: int, node_type: str):
        # sourcery skip: assign-if-exp, switch
        if self.nodes[node_ind * 2] is None:
            next_pos = node_ind * 2
        else:
            next_pos = node_ind * 2 + 1

        if node_type == "leaf":
            if next_pos % 2 == 1 and self.nodes[next_pos - 1].is_leaf():
                new_node = Node.leaf(random.choice([i for i in labels if i != self.nodes[next_pos - 1].label]))
            else:
                new_node = Node.leaf(random.choice(labels))
        elif node_type == "split":
            feature = random.choice(features)
            new_node = Node.split(feature, random.choice(feature_vals[feature]))

        self.nodes[next_pos] = new_node
        self.depth_l[next_pos] = self.depth_l[node_ind] + 1
        self.depth = max(self.depth, self.depth_l[next_pos])
        return next_pos

    def classify_one(self, X: np.ndarray):
        cur = 1
        while True:
            node = self.nodes[cur]
            if node.label is not None:
                return node.label

            cur *= 2
            if X[node.feature] > node.split_val:
                cur += 1

    def classify_many(self, X: np.ndarray):
        return [self.classify_one(x) for x in X]

    def accuracy(self, X: np.ndarray, Y: np.ndarray):
        predictions = np.array(self.classify_many(X), dtype=int)
        return np.mean(predictions == Y)

    def visualize(self):
        raise NotImplementedError("ok")


class FitnessEvaluator:
    def __init__(self, a1: float, a2: float, max_depth: int):
        self.a1 = a1
        self.a2 = a2
        self.max_depth = max_depth

    def __call__(self, individual: DecisionTree, X: np.ndarray, Y: np.ndarray):
        predictions = np.array(individual.classify_many(X), dtype=int)
        f1 = np.sum(predictions == Y) / len(Y)
        f2 = 0  # TODO: figure out how f2 works
        return self.a1 * f1 + self.a2 * f2


def selection(population: List[DecisionTree], fitness: List[float], k: int):
    # tournament selection
    assert len(population) == len(fitness)

    inds = random.sample(range(len(population)), k)
    ind = max(inds, key=lambda i: fitness[i])
    p1 = population[ind]

    inds = random.sample(range(len(population)), k)
    ind = max(inds, key=lambda i: fitness[i])
    p2 = population[ind]

    return p1, p2


def crossover(p1: DecisionTree, p2: DecisionTree):
    """
    take two parents
    child 1 = copy of parent 1
        select random node from p1 and p2
        replace node from p1 with that from p2
    child 2 = copy of parent 2
        select random node from p1 and p2
        replace node from p2 with that from p1
    """

    def replace(source: DecisionTree, dest: DecisionTree, ind: int):
        q = deque([ind])
        while q:
            cur = q.popleft()
            dest.nodes[cur] = source.nodes[cur]
            if dest.nodes[cur].label is None:
                q.append(cur * 2)
                q.append(cur * 2 + 1)

        # clean unused nodes
        # let garbage collector do the heavy lifting
        for i in range(2, len(dest.nodes)):
            if dest.nodes[i // 2] is None:
                dest.nodes[i] = None

    overlaps = [
        i
        for i in range(len(p1.nodes))
        if p1.nodes[i] is not None and p2.nodes[i] is not None
    ]

    c1 = copy.deepcopy(p1)
    ind = random.choice(overlaps)
    replace(p2, c1, ind)

    c2 = copy.deepcopy(p2)
    ind = random.choice(overlaps)
    replace(p1, c2, ind)

    return c1, c2


def crossover_v2(p1: DecisionTree, p2: DecisionTree):
    """
    a more random and less algoritmic crossover
    takes into consideration all nodes rather
    than just the ones on the same depth
    """

    def replace(
        source: DecisionTree, source_ind: int, dest: DecisionTree, dest_ind: int
    ):
        print(source_ind, dest_ind)
        q = deque([(source_ind, dest_ind)])
        while q:
            si, di = q.popleft()
            dest.nodes[di] = copy.deepcopy(source.nodes[si])
            if not source.nodes[si].is_leaf():
                q.append((si * 2, di * 2))
                q.append((si * 2 + 1, di * 2 + 1))

        # clean unused nodes
        reachable = [False for i in range(len(dest.nodes))]
        q = deque([dest_ind])
        reachable[dest_ind] = True
        while q:
            cur = q.popleft()
            if not reachable[cur]:
                dest.nodes[cur] = None
            elif not dest.nodes[cur].is_leaf() and cur * 2 < len(dest.nodes):
                reachable[cur * 2] = True
                reachable[cur * 2 + 1] = True
            if cur * 2 < len(dest.nodes):
                q.append(cur * 2)
                q.append(cur * 2 + 1)

    p1_inds = [i for i in range(len(p1.nodes)) if p1.nodes[i] is not None]
    p2_inds = [i for i in range(len(p2.nodes)) if p2.nodes[i] is not None]

    c1 = copy.deepcopy(p1)
    replace(p2, random.choice(p2_inds), c1, random.choice(p1_inds))
    c2 = copy.deepcopy(p2)
    replace(p1, random.choice(p1_inds), c2, random.choice(p2_inds))
    return c1, c2


def mutate(tree: DecisionTree):
    valid = [i for i in range(len(tree.nodes)) if tree.nodes[i] is not None]
    ind = random.choice(valid)
    if tree.nodes[ind].label is None:
        feature = random.choice(features)
        tree.nodes[ind] = Node.split(feature, random.choice(feature_vals[feature]))
    else:
        tree.nodes[ind] = Node.leaf(random.choice(labels))


class EDTClassifier:
    def __init__(
        self,
        population_size: int,
        split_proba: float,
        crossover_probability: float,
        mutation_probability: float,
        max_depth: int,
        fitness_evaluator: FitnessEvaluator,
    ):
        assert data_loaded, "Training data not loaded"

        self.cross_p = crossover_probability
        self.mut_p = mutation_probability
        self.max_depth = max_depth
        self.fitness_eval = fitness_evaluator

        # initiate population
        self.population = [
            DecisionTree.generate_random(max_depth, split_proba)
            for _ in range(population_size)
        ]

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        generations: int,
        verbose: bool = False,
    ):
        fit_eval = self.fitness_eval

        n = len(self.population)

        for gen in range(1, generations + 1):
            if verbose:
                print(f"Generation {gen}/{generations}")

            fitnesses = [fit_eval(tree, X, Y) for tree in self.population]

            if verbose:
                print(f"Average fitness: {sum(fitnesses) / len(fitnesses)}")

            new_pop = []

            # tournament selection + crossover
            for _ in range(int(n * self.cross_p / 2)):
                p1, p2 = selection(self.population, fitnesses, 2)
                c1, c2 = crossover(p1, p2)
                new_pop.extend((c1, c2))

            # elitism
            # fill new population with best individuals fom previous generation
            fp = sorted(
                zip(fitnesses, self.population), key=lambda x: x[0], reverse=True
            )
            new_pop.extend(fp[i][1] for i in range(n - len(new_pop)))

            # mutation
            for i in random.sample(range(n), int(n * self.mut_p)):
                mutate(new_pop[i])

            # TODO: delete old gen if this takes too much memory
            self.population = new_pop

            if verbose:
                print()

        # sort by fitness
        fitnesses = [fit_eval(tree, X, Y) for tree in self.population]
        fp = sorted(zip(fitnesses, self.population), key=lambda x: x[0], reverse=True)
        self.population = [fp[i][1] for i in range(n)]

    def predict(self, X: np.ndarray, top_k: float = 1):
        assert 0 < top_k <= 1

        predictions = [
            tree.classify_many(X)
            for tree in self.population[: int(top_k * len(self.population))]
        ]

        return [statistics.mode(individidual) for individidual in zip(*predictions)]
