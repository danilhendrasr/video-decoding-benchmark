import matplotlib.pyplot as plt
from typing import TypedDict
from statistics import median, stdev
from numpy import percentile
import matplotlib
matplotlib.use("Agg")


class SummaryObject(TypedDict):
    avg: "list[float]"
    min: "list[float]"
    max: "list[float]"


class IterationResult(TypedDict):
    fpt: "list[float]"
    cpu: "list[float]"
    mem: "list[float]"
    gpu: "list[float]"
    gpu_mem: "list[float]"


class BenchmarkResult:
    def __init__(self) -> None:
        self.frame_processing_times = []
        self.cpu_utils = []
        self.mem_utils = []
        self.gpu_utils = []
        self.gpu_mem_utils = []

    def summarize_fpt(self) -> "tuple[float, float, float]":
        self.frame_processing_times.sort()
        average = round(avg(self.frame_processing_times), 2)
        minimum = round(min(self.frame_processing_times), 2)
        maximum = round(max(self.frame_processing_times), 2)
        q1_percentile = round(percentile(self.frame_processing_times, 25), 2)
        q2_percentile = round(percentile(self.frame_processing_times, 50), 2)
        q3_percentile = round(percentile(self.frame_processing_times, 75), 2)
        std_deviation = round(stdev(self.frame_processing_times), 2)

        return (average, minimum, maximum, q1_percentile, q2_percentile, q3_percentile, std_deviation)

    def summarize_cpu_utils(self) -> "tuple[float, float, float]":
        self.cpu_utils.sort()
        average = round(avg(self.cpu_utils), 2)
        minimum = round(min(self.cpu_utils), 2)
        maximum = round(max(self.cpu_utils), 2)
        q1_percentile = round(percentile(self.cpu_utils, 25), 2)
        q2_percentile = round(percentile(self.cpu_utils, 50), 2)
        q3_percentile = round(percentile(self.cpu_utils, 75), 2)
        std_deviation = round(stdev(self.cpu_utils), 2)

        return (average, minimum, maximum, q1_percentile, q2_percentile, q3_percentile, std_deviation)

    def summarize_mem_utils(self) -> "tuple[float, float, float]":
        self.mem_utils.sort()
        average = round(avg(self.mem_utils), 2)
        minimum = round(min(self.mem_utils), 2)
        maximum = round(max(self.mem_utils), 2)
        q1_percentile = round(percentile(self.mem_utils, 25), 2)
        q2_percentile = round(percentile(self.mem_utils, 50), 2)
        q3_percentile = round(percentile(self.mem_utils, 75), 2)
        std_deviation = round(stdev(self.mem_utils), 2)

        return (average, minimum, maximum, q1_percentile, q2_percentile, q3_percentile, std_deviation)

    def summarize_gpu_utils(self) -> "tuple[float, float, float]":
        self.gpu_utils.sort()
        average = round(avg(self.gpu_utils), 2)
        minimum = round(min(self.gpu_utils), 2)
        maximum = round(max(self.gpu_utils), 2)
        q1_percentile = round(percentile(self.gpu_utils, 25), 2)
        q2_percentile = round(percentile(self.gpu_utils, 50), 2)
        q3_percentile = round(percentile(self.gpu_utils, 75), 2)
        std_deviation = round(stdev(self.gpu_utils), 2)

        return (average, minimum, maximum, q1_percentile, q2_percentile, q3_percentile, std_deviation)

    def summarize_gpu_mem_utils(self) -> "tuple[float, float, float]":
        self.gpu_mem_utils.sort()
        average = round(avg(self.gpu_mem_utils), 2)
        minimum = round(min(self.gpu_mem_utils), 2)
        maximum = round(max(self.gpu_mem_utils), 2)
        q1_percentile = round(percentile(self.gpu_mem_utils, 25), 2)
        q2_percentile = round(percentile(self.gpu_mem_utils, 50), 2)
        q3_percentile = round(percentile(self.gpu_mem_utils, 75), 2)
        std_deviation = round(stdev(self.gpu_mem_utils), 2)

        return (average, minimum, maximum, q1_percentile, q2_percentile, q3_percentile, std_deviation)


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


def store_benchmark_summary(iteration_result: IterationResult, storage: BenchmarkResult):
    storage.frame_processing_times = iteration_result["processing_time"]
    storage.cpu_utils = iteration_result["cpu"]
    storage.mem_utils = iteration_result["mem"]
    storage.gpu_utils = iteration_result["gpu"]
    storage.gpu_mem_utils = iteration_result["gpu_mem"]
