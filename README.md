# Video Decoding Benchmark
A program to benchmark the performance of NVIDIA NVDEC (through VideoProcessingFramework), PyAV, and OpenCV (without hardware acceleration) on video decoding operation. The following metrics are observed:
1. Frame Processing Time (the time it takes to decode 1 frame)
2. CPU Utilization Across All Cores
3. Memory Utilization
4. GPU Utilization
5. GPU Memory Utilization

## Prerequisites
- Docker
- NVIDIA Video Codec SDK

## Running the Benchmark
1. Clone the repo
2. Build the docker image
   ```bash
   docker build -t videc-benchmark .
   ```
3. Run the docker image
   ```bash
   docker run --gpus all -it videc-benchmark bash
   ```
4. Enter the following command inside the container's terminal
   ```bash
   python3 ./install/bin/main.py ./videos/5-minutes.mp4 $warmup_iteration
   ```
   Notes:
   - You should replace `$warmup_iteration` with any integer. The program will use
    the first `$warmup_iteration` amount of frames as warmup during the benchmark.
   - The `./videos/5-minutes.mp4` part is the path to input file. You can look into
    the `videos` directory to see files that are available to be used as input.
5. The benchmark result will get written to the `benchmark-results` directory 
  inside the container, you can use [docker cp](https://stackoverflow.com/a/22050116)
  to copy it to your machine.
  