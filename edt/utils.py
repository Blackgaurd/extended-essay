# most performant functions seem to be either
# negative_quadratic
#

import math


def linear(depth: int, optimal_depth: int) -> float:
    return 1 - depth / optimal_depth


def tanh(depth: int, optimal_depth: int) -> float:
    return math.tanh(linear(depth, optimal_depth))


def sigmoid(depth: int, optimal_depth: int) -> float:
    return 1 / (1 + math.exp(depth - optimal_depth)) - 0.5


def f(depth: int, optimal_depth: int) -> float:
    return min(0, linear(depth, optimal_depth))
