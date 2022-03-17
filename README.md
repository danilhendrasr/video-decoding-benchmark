# Video Decoding Benchmark
## Prerequisites
- Python 3 (I use v3.8.10)
- GCC (I use v9.3.0)
- Linux (I use Ubuntu v18.04)
- NVIDIA Cuda Toolkit (I use v11.1)
- NVIDIA Video Codec SDK (I use v11.1.5)
- FFMPEG (I use v4.2.4)

## Running the benchmark
1. Clone the repo
2. Make sure you've already set `PATH_TO_SDK` and `CUDACXX` environment variable.
  - `PATH_TO_SDK` is the path to NVIDIA Video Codec SDK
  - `CUDACXX` is the path to NVIDIA CUDA Toolkit's `nvcc` binary
3. Give execute permission for scripts inside the `scripts` directory:
```bash
sudo chmod u+x scripts/*
```
4. Run the `run-benchmark.sh`, `fetch-videos.sh`, and `build.sh` scripts by typing the following command
```bash
scripts/run-benchmark.sh -bv
```
The above command will build the project, fetch a sample video from a source, create 5 new directories at the project root, and then run the benchmark.

As for the 5 new directories that are created at the project root are:
  - `videos` contains the video that you can use as a sample to run the benchmark on
  - `videc_benchmark_env` the Python virtual environment
  - `install` contains the runnable Python program with a PyNvCodec build
  - `build` which is the CMake build directory
  - `benchmark-results` which is used to save the benchmark results from plots to markdown report
