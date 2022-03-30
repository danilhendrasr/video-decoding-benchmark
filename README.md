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
2. Download [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download) and place the zip file in the root directory of this project.
3. Build the docker image
   ```bash
   docker build -t videc-benchmark .
   ```
   2 build args are available:
   - `PROJECT_PATH`, to control where in the image you want this project to be copied to, this arg defaults to `/videc-benchmark`.
   - `VIDEO_CODEC_SDK_VERSION`, to specify which version of NVIDIA Video Codec SDK that you've downloaded from step 2, this arg defaults to 11.1.5.
3. Run the docker image
   ```bash
   docker run --gpus all -it --pid host videc-benchmark bash
   ```
4. Enter the following command inside the container's terminal
   ```bash
   python3 ./install/bin/main.py ./videos/45-seconds.mp4 $x
   ```
   Notes:
   - You should replace `$x` with an integer. The program will use
    the first `$x` frames as warmup during the benchmark, meaning it won't collect any data during the processing of the first `$x` frames.
   - The `./videos/45-seconds.mp4` part is the path to input file. You can look into
    the `videos` directory to see files that are available to be used as input.
5. The benchmark result will get written to the `benchmark-results` directory 
  inside the container, you can use [docker cp](https://stackoverflow.com/a/22050116)
  to copy it to your machine.
  
