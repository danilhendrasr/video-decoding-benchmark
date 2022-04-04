import time
import sys
import setproctitle
from tools import PyAV, OpenCV, NVDec

PROCESS_NAME = "videc-benchmark"


if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print("Provide tool to use, input file, and warm up frames")
        exit(1)

    tool_to_run = sys.argv[1]
    file_to_decode = sys.argv[2]
    warmup_frames = int(sys.argv[3])
    setproctitle.setproctitle(PROCESS_NAME)

    if tool_to_run == "all":
        opencv = OpenCV(file_to_decode, True, PROCESS_NAME)
        opencv.decode(warmup_iteration=warmup_frames)
        opencv_summary = opencv.summarize_records()
        opencv.summary_to_csv("OpenCV", opencv_summary)

        time.sleep(5)

        pyav = PyAV(file_to_decode, True, PROCESS_NAME)
        pyav.decode(warmup_iteration=warmup_frames)
        pyav_summary = pyav.summarize_records()
        pyav.summary_to_csv("PyAV", pyav_summary)

        time.sleep(5)

        nvdec = NVDec(file_to_decode, True, PROCESS_NAME)
        nvdec.decode(warmup_iteration=warmup_frames)
        nvdec_summary = nvdec.summarize_records()
        nvdec.summary_to_csv("NVDEC", nvdec_summary)

    elif tool_to_run == "pyav":
        pyav = PyAV(file_to_decode, True, PROCESS_NAME)
        pyav.decode(warmup_iteration=warmup_frames)
        pyav_summary = pyav.summarize_records()
        pyav.summary_to_csv("PyAV", pyav_summary)

    elif tool_to_run == "opencv":
        opencv = OpenCV(file_to_decode, True, PROCESS_NAME)
        opencv.decode(warmup_iteration=warmup_frames)
        opencv_summary = opencv.summarize_records()
        opencv.summary_to_csv("OpenCV", opencv_summary)

    elif tool_to_run == "nvdec":
        nvdec = NVDec(file_to_decode, True, PROCESS_NAME)
        nvdec.decode(warmup_iteration=warmup_frames)
        nvdec_summary = nvdec.summarize_records()
        nvdec.summary_to_csv("NVDEC", nvdec_summary)

    else:
        print("Can only run the following tools: all, nvdec, pyav, opencv")
        exit(1)

    exit(0)
