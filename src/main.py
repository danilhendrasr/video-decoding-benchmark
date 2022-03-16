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
import setproctitle

nvidia_smi.nvmlInit()

PROCESS_NAME = "videc-benchmark"
PROCESS_PID = 0

setproctitle.setproctitle(PROCESS_NAME)


def measure_decode_with_nvdec(gpu_id: int, file_to_decode: str, current_iteration: int) -> IterationResult:
    dec = NvDecoder(gpu_id, file_to_decode)
    decode_result = dec.decode(
        dump_frames=False, current_iteration=current_iteration)
    return decode_result


def measure_decode_with_pyav(file_to_decode: str, current_iteration: int) -> IterationResult:
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

    av_input = av.open(file_to_decode)
    for packet in av_input.demux():
        if packet.size == 0:
            continue

        gpu = gpustat.core.GPUStatCollection.new_query()
        start_counter = time.perf_counter()
        packet.decode()
        end_counter = time.perf_counter()

        processing_time = s_to_ms(end_counter - start_counter)
        cpu_util = psutil_handle.cpu_percent()
        mem_util = utils.b_to_mb(psutil_handle.memory_info().rss)
        gpu_util = gpu[0].utilization
        gpu_mem_util = gpu[0].memory_used
        # gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(gpu_handle)

        frame_decode_record.append(processing_time)
        cpu_util_record.append(cpu_util)
        mem_util_record.append(mem_util)
        gpu_util_record.append(gpu_util)
        gpu_mem_util_record.append(gpu_mem_util)

    plot_list_to_image(
        cpu_util_record, 'benchmark-results/plot/cpu/pyav-cpu-{}.png'.format(current_iteration))

    return {
        "processing_time": frame_decode_record,
        "cpu": cpu_util_record,
        "mem": mem_util_record,
        "gpu": gpu_util_record,
        "gpu_mem": gpu_mem_util_record
    }


def measure_decode_with_opencv(file_to_decode: str, current_iteration: int) -> IterationResult:
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

        start_counter = time.perf_counter()
        ret, _ = video.read()
        if not ret:
            break
        end_counter = time.perf_counter()

        processing_time = s_to_ms(end_counter - start_counter)
        cpu_util = psutil_handle.cpu_percent()
        mem_util = utils.b_to_mb(psutil_handle.memory_info().rss)
        gpu_util = gpu[0].utilization
        gpu_mem_util = gpu[0].memory_used
        # gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(gpu_handle)

        frame_decode_record.append(processing_time)
        cpu_util_record.append(cpu_util)
        mem_util_record.append(mem_util)
        gpu_util_record.append(gpu_util)
        gpu_mem_util_record.append(gpu_mem_util)

    video.release()

    plot_list_to_image(
        cpu_util_record, 'benchmark-results/plot/cpu/opencv-cpu-{}.png'.format(current_iteration))

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

    time.sleep(5)

    print("Running benchmark...")

    nvcuvid_result = measure_decode_with_nvdec(
        gpu_id, file_to_decode, nth_iteration)
    store_benchmark_summary(nvcuvid_result, nvdec_benchmark_results)

    time.sleep(3)

    pyav_result = measure_decode_with_pyav(
        file_to_decode, nth_iteration)
    store_benchmark_summary(pyav_result, pyav_benchmark_results)

    time.sleep(3)

    opencv_result = measure_decode_with_opencv(
        file_to_decode, nth_iteration)
    store_benchmark_summary(opencv_result, opencv_benchmark_results)

    results_table = {
        "Frame Processing Time (ms)": [
            ["NVDEC", *nvdec_benchmark_results.summarize_fpt()],
            ["PyAV", *pyav_benchmark_results.summarize_fpt()],
            ["OpenCV", *opencv_benchmark_results.summarize_fpt()]
        ],
        "CPU Utilization (%)": [
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

    result_markdown_path = '{}/benchmark-results/results.md'.format(
        os.getcwd())
    result_md = open(result_markdown_path, 'w')

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

    print("Benchmark result written to: {}".format(result_markdown_path))

    exit(0)
