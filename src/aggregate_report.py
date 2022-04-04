import os
import csv

result_markdown_path = '{}/benchmark-results/report.md'.format(
    os.getcwd())
result_md = open(result_markdown_path, 'w')

results_table = {
    "Frame Processing Time (ms)": [],
    "CPU Utilization Across All Cores (%)": [],
    "Memory Utilization (MB)": [],
    "GPU Utilization (%)": [],
    "GPU Memory Utilization (MB)": [],
}

csv_files = os.listdir('benchmark-results/individual_summary')
for file in csv_files:
    if file.endswith(".csv"):
        with open('benchmark-results/individual_summary/{}'.format(file), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            result_table_keys = list(results_table.keys())
            for i, row in enumerate(csv_reader):
                results_table[result_table_keys[i]].append(
                    [file.split(".")[0], *row])

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
