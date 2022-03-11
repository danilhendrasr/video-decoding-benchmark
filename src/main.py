import time
import av
import sys
from nvdecoder import NvDecoder
from tqdm import tqdm
import cv2
import psutil
import nvidia_smi
from utils import s_to_ms, plot_list_to_image, list_summary, avg, store_benchmark_summary, BenchmarkResult, IterationResult
import utils
from prettytable import PrettyTable

nvidia_smi.nvmlInit()


def measure_decode_with_nvcuvid(gpu_id: int, file_to_decode: str, current_iteration: int) -> IterationResult:
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

    gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    av_input = av.open(file_to_decode)
    for packet in av_input.demux():
        if packet.size == 0:
            continue

        start_counter = time.perf_counter()
        packet.decode()
        end_counter = time.perf_counter()

        processing_time = s_to_ms(end_counter - start_counter)
        cpu_util = psutil.cpu_percent()
        mem_util = utils.b_to_mb(psutil.virtual_memory().used)
        gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(gpu_handle)

        frame_decode_record.append(processing_time)
        cpu_util_record.append(cpu_util)
        mem_util_record.append(mem_util)
        gpu_util_record.append(gpu_util.gpu)
        gpu_mem_util_record.append(gpu_util.memory)

    plot_list_to_image(
        cpu_util_record, 'benchmark-results/plot/cpu/pyav-cpu-{}.png'.format(current_iteration))

    processing_time_summary = list_summary(frame_decode_record)
    cpu_util_summary = list_summary(cpu_util_record)
    mem_util_summary = list_summary(mem_util_record)
    gpu_util_summary = list_summary(gpu_util_record)
    gpu_mem_util_summary = list_summary(gpu_mem_util_record)

    return {
        "processing_time": {
            "avg": processing_time_summary[0],
            "min": processing_time_summary[1],
            "max": processing_time_summary[2],
        },
        "cpu": {
            "avg": cpu_util_summary[0],
            "min": cpu_util_summary[1],
            "max": cpu_util_summary[2],
        },
        "mem": {
            "avg": mem_util_summary[0],
            "min": mem_util_summary[1],
            "max": mem_util_summary[2],
        },
        "gpu": {
            "avg": gpu_util_summary[0],
            "min": gpu_util_summary[1],
            "max": gpu_util_summary[2],
        },
        "gpu_mem": {
            "avg": gpu_mem_util_summary[0],
            "min": gpu_mem_util_summary[1],
            "max": gpu_mem_util_summary[2],
        }
    }


def measure_decode_with_opencv(file_to_decode: str, current_iteration: int) -> IterationResult:
    frame_decode_record = []
    cpu_util_record = []
    mem_util_record = []
    gpu_util_record = []
    gpu_mem_util_record = []

    gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    video = cv2.VideoCapture(file_to_decode)
    while video.isOpened():
        start_counter = time.perf_counter()
        ret, _ = video.read()
        if not ret:
            break
        end_counter = time.perf_counter()

        processing_time = s_to_ms(end_counter - start_counter)
        cpu_util = psutil.cpu_percent()
        mem_util = utils.b_to_mb(psutil.virtual_memory().used)
        gpu_util = nvidia_smi.nvmlDeviceGetUtilizationRates(gpu_handle)

        frame_decode_record.append(processing_time)
        cpu_util_record.append(cpu_util)
        mem_util_record.append(mem_util)
        gpu_util_record.append(gpu_util.gpu)
        gpu_mem_util_record.append(gpu_util.memory)

    video.release()

    plot_list_to_image(
        cpu_util_record, 'benchmark-results/plot/cpu/opencv-cpu-{}.png'.format(current_iteration))

    processing_time_summary = list_summary(frame_decode_record)
    cpu_util_summary = list_summary(cpu_util_record)
    mem_util_summary = list_summary(mem_util_record)
    gpu_util_summary = list_summary(gpu_util_record)
    gpu_mem_util_summary = list_summary(gpu_mem_util_record)

    return {
        "processing_time": {
            "avg": processing_time_summary[0],
            "min": processing_time_summary[1],
            "max": processing_time_summary[2],
        },
        "cpu": {
            "avg": cpu_util_summary[0],
            "min": cpu_util_summary[1],
            "max": cpu_util_summary[2],
        },
        "mem": {
            "avg": mem_util_summary[0],
            "min": mem_util_summary[1],
            "max": mem_util_summary[2],
        },
        "gpu": {
            "avg": gpu_util_summary[0],
            "min": gpu_util_summary[1],
            "max": gpu_util_summary[2],
        },
        "gpu_mem": {
            "avg": gpu_mem_util_summary[0],
            "min": gpu_mem_util_summary[1],
            "max": gpu_mem_util_summary[2],
        }
    }


if __name__ == "__main__":

    print("This sample decodes input video to raw NV12 file on given GPU.")
    print("Usage: SampleDecode.py $gpu_id $input_file.")

    if(len(sys.argv) < 3):
        print("Provide gpu ID, path to input and output files")
        exit(1)

    gpu_id = int(sys.argv[1])
    file_to_decode = sys.argv[2]
    iteration_count = 1

    nvcuvid_benchmark_results = BenchmarkResult()
    pyav_benchmark_results = BenchmarkResult()
    opencv_benchmark_results = BenchmarkResult()

    print("Running benchmark...")
    with tqdm(range(iteration_count)) as t:
        for i in t:
            nth_iteration = i + 1
            tqdm.write(
                "---- Iteration {}/{} ----".format(i+1, iteration_count))

            t.set_description("Running NVCUVID")
            nvcuvid_result = measure_decode_with_nvcuvid(
                gpu_id, file_to_decode, nth_iteration)

            store_benchmark_summary(
                nvcuvid_result, nvcuvid_benchmark_results)
            t.set_description("NVCUVID finished")

            time.sleep(10)

            t.set_description("Running PyAV")
            pyav_result = measure_decode_with_pyav(
                file_to_decode, nth_iteration)

            store_benchmark_summary(
                pyav_result, pyav_benchmark_results)
            t.set_description("PyAV finished")

            time.sleep(10)

            t.set_description("Running OpenCV")
            opencv_result = measure_decode_with_opencv(
                file_to_decode, nth_iteration)

            store_benchmark_summary(
                opencv_result, opencv_benchmark_results)
            t.set_description("OpenCV finished")

            t.set_description("Finished benchmarking")

    fpt_result_table = [
        [
            "NVCUVID",
            *nvcuvid_benchmark_results.summarize_fpt(),
        ],
        [
            "PyAV",
            *pyav_benchmark_results.summarize_fpt(),
        ],
        [
            "OpenCV",
            *opencv_benchmark_results.summarize_fpt(),
        ]
    ]

    cpu_utils_result_table = [
        [
            "NVCUVID",
            *nvcuvid_benchmark_results.summarize_cpu_utils(),
        ],
        [
            "PyAV",
            *pyav_benchmark_results.summarize_cpu_utils(),
        ],
        [
            "OpenCV",
            *opencv_benchmark_results.summarize_cpu_utils(),
        ]
    ]

    mem_utils_result_table = [
        [
            "NVCUVID",
            *nvcuvid_benchmark_results.summarize_mem_utils(),
        ],
        [
            "PyAV",
            *pyav_benchmark_results.summarize_mem_utils(),
        ],
        [
            "OpenCV",
            *opencv_benchmark_results.summarize_mem_utils(),
        ]
    ]

    gpu_utils_result_table = [
        [
            "NVCUVID",
            *nvcuvid_benchmark_results.summarize_gpu_utils(),
        ],
        [
            "PyAV",
            *pyav_benchmark_results.summarize_gpu_utils(),
        ],
        [
            "OpenCV",
            *opencv_benchmark_results.summarize_gpu_utils(),
        ]
    ]

    gpu_mem_utils_result_table = [
        [
            "NVCUVID",
            *nvcuvid_benchmark_results.summarize_gpu_mem_utils(),
        ],
        [
            "PyAV",
            *pyav_benchmark_results.summarize_gpu_mem_utils(),
        ],
        [
            "OpenCV",
            *opencv_benchmark_results.summarize_gpu_mem_utils(),
        ]
    ]

    print("\n---- Benchmark Result ----")

    print("Case      : Measure time to decode each frame in %s" %
          file_to_decode)
    print("Iteration : %d" % iteration_count)

    fpt_table = PrettyTable()
    fpt_table.title = "Frame Processing Time (in milliseconds)"
    fpt_table.field_names = ["Tool", "Mean (ms)", "Min (ms)", "Max (ms)"]
    for i in fpt_result_table:
        fpt_table.add_row(i)
    print(fpt_table)

    cpu_utils_table = PrettyTable()
    cpu_utils_table.title = "CPU Utilization Across All Cores (in percent)"
    cpu_utils_table.field_names = ["Tool", "Mean (%)", "Min (%)", "Max (%)"]
    for i in cpu_utils_result_table:
        cpu_utils_table.add_row(i)
    print(cpu_utils_table)

    mem_utils_table = PrettyTable()
    mem_utils_table.title = "Memory Utilization (in MB)"
    mem_utils_table.field_names = ["Tool", "Mean (MB)", "Min (MB)", "Max (MB)"]
    for i in mem_utils_result_table:
        mem_utils_table.add_row(i)
    print(mem_utils_table)

    gpu_utils_table = PrettyTable()
    gpu_utils_table.title = "GPU Utilization (in percent)"
    gpu_utils_table.field_names = ["Tool", "Mean (%)", "Min (%)", "Max (%)"]
    for i in gpu_utils_result_table:
        gpu_utils_table.add_row(i)
    print(gpu_utils_table)

    gpu_mem_utils_table = PrettyTable()
    gpu_mem_utils_table.title = "GPU Memory Utilization (in percent)"
    gpu_mem_utils_table.field_names = [
        "Tool", "Mean (%)", "Min (%)", "Max (%)"]
    for i in gpu_mem_utils_result_table:
        gpu_mem_utils_table.add_row(i)
    print(gpu_mem_utils_table)

    exit(0)
