import re
import time
import av
import sys
from nvdecoder import NvDecoder
from tqdm import tqdm
import cv2
import psutil
import nvidia_smi
import gpustat
from utils import s_to_ms, plot_list_to_image, store_benchmark_summary, BenchmarkResult, IterationResult
import utils
import os
import csv
import setproctitle

nvidia_smi.nvmlInit()

PROCESS_NAME = "videc-benchmark"
PROCESS_PID = 0

setproctitle.setproctitle(PROCESS_NAME)


def measure_decode_with_nvdec(gpu_id: int, file_to_decode: str, with_plot=False) -> IterationResult:
    dec = NvDecoder(gpu_id, file_to_decode)
    decode_result = dec.decode(
        dump_frames=False, with_plot=with_plot)
    return decode_result


def measure_decode_with_pyav(file_to_decode: str, with_plot=False) -> IterationResult:
    frame_decode_record = []
    cpu_util_record = []
    mem_util_record = []
    gpu_util_record = []
    gpu_mem_util_record = []

    process_name = re.compile(PROCESS_NAME)
    for p in psutil.process_iter(['pid', 'name', 'memory_info']):
        if not process_name.match(p.name()):
            continue
        PROCESS_PID = p.pid

    psutil_handle = psutil.Process(PROCESS_PID)

    av_input = av.open(file_to_decode)
    for packet in av_input.demux():
        if packet.size == 0:
            continue

        gpu = gpustat.core.GPUStatCollection.new_query()
        gpu_processes = filter(lambda x: process_name.match(
            x['command']), gpu[0].processes)
        start_counter = time.perf_counter()
        packet.decode()
        end_counter = time.perf_counter()

        processing_time = round(s_to_ms(end_counter - start_counter), 2)
        cpu_util = psutil.cpu_percent()
        mem_util = round(utils.b_to_mb(psutil_handle.memory_info().rss), 2)
        gpu_util = gpu[0].utilization

        for process in gpu_processes:
            gpu_mem_util_record.append(process["gpu_memory_usage"])

        frame_decode_record.append(processing_time)
        cpu_util_record.append(cpu_util)
        mem_util_record.append(mem_util)
        gpu_util_record.append(gpu_util)

    # Dump records to CSV files
    with open("benchmark-results/csv/fpt/pyav.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(frame_decode_record)
    with open("benchmark-results/csv/cpu/pyav.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(cpu_util_record)
    with open("benchmark-results/csv/mem/pyav.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(mem_util_record)
    with open("benchmark-results/csv/gpu/pyav.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(gpu_util_record)
    with open("benchmark-results/csv/gpu-mem/pyav.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(gpu_mem_util_record)

    if with_plot:
        plot_list_to_image(
            frame_decode_record, 'benchmark-results/plot/fpt/pyav.png')
        plot_list_to_image(
            cpu_util_record, 'benchmark-results/plot/cpu/pyav.png')
        plot_list_to_image(
            mem_util_record, 'benchmark-results/plot/mem/pyav.png')
        plot_list_to_image(
            gpu_util_record, 'benchmark-results/plot/gpu/pyav.png')
        plot_list_to_image(
            gpu_mem_util_record, 'benchmark-results/plot/gpu-mem/pyav.png')

    return {
        "processing_time": frame_decode_record,
        "cpu": cpu_util_record,
        "mem": mem_util_record,
        "gpu": gpu_util_record,
        "gpu_mem": gpu_mem_util_record
    }


def measure_decode_with_opencv(file_to_decode: str, with_plot=False) -> IterationResult:
    frame_decode_record = []
    cpu_util_record = []
    mem_util_record = []
    gpu_util_record = []
    gpu_mem_util_record = []

    process_name = re.compile(PROCESS_NAME)
    for p in psutil.process_iter(['pid', 'name', 'memory_info']):
        if not process_name.match(p.name()):
            continue
        PROCESS_PID = p.pid

    # gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    psutil_handle = psutil.Process(PROCESS_PID)

    video = cv2.VideoCapture(file_to_decode)
    while video.isOpened():
        gpu = gpustat.core.GPUStatCollection.new_query()
        gpu_processes = filter(lambda x: process_name.match(
            x['command']), gpu[0].processes)

        start_counter = time.perf_counter()
        ret, _ = video.read()
        if not ret:
            break
        end_counter = time.perf_counter()

        processing_time = round(s_to_ms(end_counter - start_counter), 2)
        cpu_util = psutil.cpu_percent()
        mem_util = round(utils.b_to_mb(psutil_handle.memory_info().rss), 2)
        gpu_util = gpu[0].utilization

        for process in gpu_processes:
            gpu_mem_util_record.append(process["gpu_memory_usage"])

        frame_decode_record.append(processing_time)
        cpu_util_record.append(cpu_util)
        mem_util_record.append(mem_util)
        gpu_util_record.append(gpu_util)

    video.release()

    # Dump records to CSV files
    with open("benchmark-results/csv/fpt/opencv.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(frame_decode_record)
    with open("benchmark-results/csv/cpu/opencv.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(cpu_util_record)
    with open("benchmark-results/csv/mem/opencv.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(mem_util_record)
    with open("benchmark-results/csv/gpu/opencv.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(gpu_util_record)
    with open("benchmark-results/csv/gpu-mem/opencv.csv", 'w') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(gpu_mem_util_record)

    if with_plot:
        plot_list_to_image(
            frame_decode_record, 'benchmark-results/plot/fpt/opencv.png')
        plot_list_to_image(
            cpu_util_record, 'benchmark-results/plot/cpu/opencv.png')
        plot_list_to_image(
            mem_util_record, 'benchmark-results/plot/mem/opencv.png')
        plot_list_to_image(
            gpu_util_record, 'benchmark-results/plot/gpu/opencv.png')
        plot_list_to_image(
            gpu_mem_util_record, 'benchmark-results/plot/gpu-mem/opencv.png')

    return {
        "processing_time": frame_decode_record,
        "cpu": cpu_util_record,
        "mem": mem_util_record,
        "gpu": gpu_util_record,
        "gpu_mem": gpu_mem_util_record
    }


if __name__ == "__main__":

    print("This sample decodes input video to raw NV12 file on given GPU.")
    print("Usage: SampleDecode.py $gpu_id $input_file.")

    if(len(sys.argv) < 3):
        print("Provide gpu ID, path to input and output files")
        exit(1)

    gpu_id = int(sys.argv[1])
    file_to_decode = sys.argv[2]
    warmup_iteration = 1

    nvdec_benchmark_results = BenchmarkResult()
    pyav_benchmark_results = BenchmarkResult()
    opencv_benchmark_results = BenchmarkResult()

    print("Running warmup...")
    with tqdm(range(warmup_iteration)) as t:
        for i in t:
            nth_iteration = i + 1
            tqdm.write("--- Iteration {}/{} ---".format(i+1, warmup_iteration))

            t.set_description("Running NVDEC")
            measure_decode_with_nvdec(gpu_id, file_to_decode, nth_iteration)
            t.set_description("NVDEC finished")

            t.set_description("Running PyAV")
            measure_decode_with_pyav(file_to_decode, nth_iteration)
            t.set_description("PyAV finished")

            t.set_description("Running OpenCV")
            measure_decode_with_opencv(file_to_decode, nth_iteration)
            t.set_description("OpenCV finished")

            t.set_description("Warmup finished")

    time.sleep(10)

    print("Running benchmark...")

    nvdec_result = measure_decode_with_nvdec(
        gpu_id, file_to_decode, with_plot=True)
    store_benchmark_summary(nvdec_result, nvdec_benchmark_results)

    time.sleep(10)

    pyav_result = measure_decode_with_pyav(
        file_to_decode, with_plot=True)
    store_benchmark_summary(pyav_result, pyav_benchmark_results)

    time.sleep(10)

    opencv_result = measure_decode_with_opencv(
        file_to_decode, with_plot=True)
    store_benchmark_summary(opencv_result, opencv_benchmark_results)

    results_table = {
        "Frame Processing Time (ms)": [
            ["NVDEC", *nvdec_benchmark_results.summarize_fpt()],
            ["PyAV", *pyav_benchmark_results.summarize_fpt()],
            ["OpenCV", *opencv_benchmark_results.summarize_fpt()]
        ],
        "CPU Utilization Across All Cores (%)": [
            ["NVDEC", *nvdec_benchmark_results.summarize_cpu_utils()],
            ["PyAV", *pyav_benchmark_results.summarize_cpu_utils()],
            ["OpenCV", *opencv_benchmark_results.summarize_cpu_utils()]
        ],
        "Memory Utilization (MB)": [
            ["NVDEC", *nvdec_benchmark_results.summarize_mem_utils()],
            ["PyAV", *pyav_benchmark_results.summarize_mem_utils()],
            ["OpenCV", *opencv_benchmark_results.summarize_mem_utils()]
        ],
        "GPU Utilization (%)": [
            ["NVDEC", *nvdec_benchmark_results.summarize_gpu_utils()],
            ["PyAV", *pyav_benchmark_results.summarize_gpu_utils()],
            ["OpenCV", *opencv_benchmark_results.summarize_gpu_utils()]
        ],
        "GPU Memory Utilization (MB)": [
            ["NVDEC", *nvdec_benchmark_results.summarize_gpu_mem_utils()],
            ["PyAV", *pyav_benchmark_results.summarize_gpu_mem_utils()],
            ["OpenCV", *opencv_benchmark_results.summarize_gpu_mem_utils()]
        ],
    }

    result_markdown_path = '{}/benchmark-results/report.md'.format(
        os.getcwd())
    result_md = open(result_markdown_path, 'w')

    result_md.write("# Benchmark Report\n")
    result_md.write("<table>")
    result_md.write("""
    <tr>
        <th colspan="8">Benchmark Results</th>
    </tr>""")
    for key, value in results_table.items():
        result_md.write("""
    <tr>
        <td colspan="8"><strong>{}</strong></td>
    </tr>
    <tr>
        <td>Tool</td>
        <td>Mean</td>
        <td>Min</td>
        <td>Max</td>
        <td>Q1</td>
        <td>Q2</td>
        <td>Q3</td>
        <td>Standard Deviation</td>
    </tr>""".format(key))
        for row in value:
            result_md.write("""
    <tr>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
    </tr>""".format(*row))

    result_md.write("\n</table>\n")
    result_md.write("\n# Plots\n")
    result_md.write("## Frame Processing Time")
    result_md.write("""
### NVDEC
![](./plot/fpt/nvdec.png)
### PyAV
![](./plot/fpt/pyav.png)
### OpenCV
![](./plot/fpt/opencv.png)\n""")
    result_md.write("## CPU Utilization")
    result_md.write("""
### NVDEC
![](./plot/cpu/nvdec.png)
### PyAV
![](./plot/cpu/pyav.png)
### OpenCV
![](./plot/cpu/opencv.png)\n""")
    result_md.write("## Memory Utilization")
    result_md.write("""
### NVDEC
![](./plot/mem/nvdec.png)
### PyAV
![](./plot/mem/pyav.png)
### OpenCV
![](./plot/mem/opencv.png)\n""")
    result_md.write("## GPU Utilization")
    result_md.write("""
### NVDEC
![](./plot/gpu/nvdec.png)
### PyAV
![](./plot/gpu/pyav.png)
### OpenCV
![](./plot/gpu/opencv.png)\n""")
    result_md.write("## GPU Memory Utilization")
    result_md.write("""
### NVDEC
![](./plot/gpu-mem/nvdec.png)
### PyAV
![](./plot/gpu-mem/pyav.png)
### OpenCV
![](./plot/gpu-mem/opencv.png)\n""")

    print("Benchmark result written to: {}".format(result_markdown_path))

    exit(0)
