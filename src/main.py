import time
import sys
from tqdm import tqdm
import os
import setproctitle
from tools import PyAV, OpenCV, NVDec

PROCESS_NAME = "videc-benchmark"

setproctitle.setproctitle(PROCESS_NAME)


if __name__ == "__main__":
    print("This program compares NVIDIA NVDEC, PyAV, and OpenCV performance on video decoding.")
    print("Usage: main.py $path_to_input_file $warmup_count")

    if(len(sys.argv) < 3):
        print("Provide input file and warmup iteration count")
        exit(1)

    file_to_decode = sys.argv[1]
    warmup_iteration = int(sys.argv[2])

    print("\nRunning benchmark...")

    pyav = PyAV(file_to_decode, True, PROCESS_NAME)
    pyav.decode(warmup_iteration=warmup_iteration)
    pyav_summary = pyav.summarize_records()

    time.sleep(5)

    opencv = OpenCV(file_to_decode, True, PROCESS_NAME)
    opencv.decode(warmup_iteration=warmup_iteration)
    opencv_summary = opencv.summarize_records()

    time.sleep(5)

    nvdec = NVDec(file_to_decode, True, PROCESS_NAME)
    nvdec.decode(warmup_iteration=warmup_iteration)
    nvdec_summary = nvdec.summarize_records()

    results_table = {
        "Frame Processing Time (ms)": [
            ["NVDEC", *nvdec_summary["fpt"].values()],
            ["PyAV", *pyav_summary["fpt"].values()],
            ["OpenCV", *opencv_summary["fpt"].values()]
        ],
        "CPU Utilization Across All Cores (%)": [
            ["NVDEC", *nvdec_summary["cpu"].values()],
            ["PyAV", *pyav_summary["cpu"].values()],
            ["OpenCV", *opencv_summary["cpu"].values()]
        ],
        "Memory Utilization (MB)": [
            ["NVDEC", *nvdec_summary["mem"].values()],
            ["PyAV", *pyav_summary["mem"].values()],
            ["OpenCV", *opencv_summary["mem"].values()]
        ],
        "GPU Utilization (%)": [
            ["NVDEC", *nvdec_summary["gpu"].values()],
            ["PyAV", *pyav_summary["gpu"].values()],
            ["OpenCV", *opencv_summary["gpu"].values()]
        ],
        "GPU Memory Utilization (MB)": [
            ["NVDEC", *nvdec_summary["gpu_mem"].values()],
            ["PyAV", *pyav_summary["gpu_mem"].values()],
            ["OpenCV", *opencv_summary["gpu_mem"].values()]
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
![](./plot/fpt/opencv.png)\n\n""")
    result_md.write("## CPU Utilization")
    result_md.write("""
### NVDEC
![](./plot/cpu/nvdec.png)
### PyAV
![](./plot/cpu/pyav.png)
### OpenCV
![](./plot/cpu/opencv.png)\n\n""")
    result_md.write("## Memory Utilization")
    result_md.write("""
### NVDEC
![](./plot/mem/nvdec.png)
### PyAV
![](./plot/mem/pyav.png)
### OpenCV
![](./plot/mem/opencv.png)\n\n""")
    result_md.write("## GPU Utilization")
    result_md.write("""
### NVDEC
![](./plot/gpu/nvdec.png)
### PyAV
![](./plot/gpu/pyav.png)
### OpenCV
![](./plot/gpu/opencv.png)\n\n""")
    result_md.write("## GPU Memory Utilization")
    result_md.write("""
### NVDEC
![](./plot/gpu_mem/nvdec.png)
### PyAV
![](./plot/gpu_mem/pyav.png)
### OpenCV
![](./plot/gpu_mem/opencv.png)\n""")

    print("\nBenchmark result written to: {}".format(result_markdown_path))

    exit(0)
