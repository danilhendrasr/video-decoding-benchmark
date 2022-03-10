import matplotlib.pyplot as plt
from typing import TypedDict
import matplotlib
matplotlib.use("Agg")

ListSummary = "tuple[float, float, float]"


class SummaryObject(TypedDict):
    avg: "list[float]"
    min: "list[float]"
    max: "list[float]"


class IterationResult(TypedDict):
    processing_time: SummaryObject
    cpu: SummaryObject
    mem: SummaryObject
    gpu: SummaryObject
    gpu_mem: SummaryObject


class BenchmarkResult:
    def __init__(self) -> None:
        self.avg_fpt = []
        self.min_fpt = []
        self.max_fpt = []

        self.avg_cpu_utils = []
        self.min_cpu_utils = []
        self.max_cpu_utils = []

        self.avg_mem_utils = []
        self.min_mem_utils = []
        self.max_mem_utils = []

        self.avg_gpu_utils = []
        self.min_gpu_utils = []
        self.max_gpu_utils = []

        self.avg_gpu_mem_utils = []
        self.min_gpu_mem_utils = []
        self.max_gpu_mem_utils = []

    def summarize_fpt(self) -> "tuple[float, float, float]":
        average = round(avg(self.avg_fpt), 2)
        minimum = round(min(self.min_fpt), 2)
        maximum = round(max(self.max_fpt), 2)

        return (average, minimum, maximum)

    def summarize_cpu_utils(self) -> "tuple[float, float, float]":
        average = round(avg(self.avg_cpu_utils), 2)
        minimum = round(min(self.min_cpu_utils), 2)
        maximum = round(max(self.max_cpu_utils), 2)

        return (average, minimum, maximum)

    def summarize_mem_utils(self) -> "tuple[float, float, float]":
        average = round(avg(self.avg_mem_utils), 2)
        minimum = round(min(self.min_mem_utils), 2)
        maximum = round(max(self.max_mem_utils), 2)

        return (average, minimum, maximum)

    def summarize_gpu_utils(self) -> "tuple[float, float, float]":
        average = round(avg(self.avg_gpu_utils), 2)
        minimum = round(min(self.min_gpu_utils), 2)
        maximum = round(max(self.max_gpu_utils), 2)

        return (average, minimum, maximum)

    def summarize_gpu_mem_utils(self) -> "tuple[float, float, float]":
        average = round(avg(self.avg_gpu_mem_utils), 2)
        minimum = round(min(self.min_gpu_mem_utils), 2)
        maximum = round(max(self.max_gpu_mem_utils), 2)

        return (average, minimum, maximum)


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
    storage.avg_fpt.append(iteration_result["processing_time"]["avg"])
    storage.min_fpt.append(iteration_result["processing_time"]["min"])
    storage.max_fpt.append(iteration_result["processing_time"]["max"])

    storage.avg_cpu_utils.append(iteration_result["cpu"]["avg"])
    storage.min_cpu_utils.append(iteration_result["cpu"]["min"])
    storage.max_cpu_utils.append(iteration_result["cpu"]["max"])

    storage.avg_mem_utils.append(iteration_result["mem"]["avg"])
    storage.min_mem_utils.append(iteration_result["mem"]["min"])
    storage.max_mem_utils.append(iteration_result["mem"]["max"])

    storage.avg_gpu_utils.append(iteration_result["gpu"]["avg"])
    storage.min_gpu_utils.append(iteration_result["gpu"]["min"])
    storage.max_gpu_utils.append(iteration_result["gpu"]["max"])

    storage.avg_gpu_mem_utils.append(iteration_result["gpu_mem"]["avg"])
    storage.min_gpu_mem_utils.append(iteration_result["gpu_mem"]["min"])
    storage.max_gpu_mem_utils.append(iteration_result["gpu_mem"]["max"])
