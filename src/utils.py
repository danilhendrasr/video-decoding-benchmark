import matplotlib.pyplot as plt
from typing import TypedDict
from statistics import median, stdev
from numpy import percentile
import matplotlib
matplotlib.use("Agg")


class IterationResult(TypedDict):
    fpt: "list[float]"
    cpu: "list[float]"
    mem: "list[float]"
    gpu: "list[float]"
    gpu_mem: "list[float]"


def plot_list_to_image(list: list, path: str):
    plt.plot(list)
    plt.savefig(path)
    plt.clf()


def list_summary(input_list: "list[int]") -> "tuple[float, float, float]":
    average = sum(input_list) / len(input_list)
    minimum = min(input_list)
    maximum = max(input_list)

    return (average, minimum, maximum)


def avg(input_list: list):
    return sum(input_list) / len(input_list)


def s_to_ms(s):
    return s * 1000


def b_to_mb(b: int):
    return b / 1_000_000
