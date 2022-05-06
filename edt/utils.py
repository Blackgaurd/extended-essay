# most performant functions seem to be either
# negative_quadratic
#

import math


def linear(depth: int, max_depth: int) -> float:
    return 1 - depth / max_depth


def log10(depth: int, max_depth: int) -> float:
    try:
        return math.log10(-depth + max_depth + 1)
    except ValueError as e:
        print(depth, max_depth)
        return float("-inf")


def piecewise_linear(depth: int, max_depth: int) -> float:
    return (
        (max_depth - depth) / (2 * max_depth)
        if depth < max_depth
        else 2 * (max_depth - depth)
    )


def piecewise_negative_linear(depth: int, max_depth: int) -> float:
    return min(0, linear(depth, max_depth))


def piecewise_negative_log10(depth: int, max_depth: int) -> float:
    return min(0, log10(depth, max_depth))


def negative_quadratic(depth: int, max_depth: int) -> float:
    return -(((depth - max_depth) / 2) ** 2)


def piecewise_negative_quadratic(depth: int, max_depth: int) -> float:
    return min(0, negative_quadratic(depth, max_depth))


def no_f2(depth: int, max_depth: int) -> float:
    return 0
