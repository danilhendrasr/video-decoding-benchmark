# Video Decoding Benchmark
A program to benchmark the performance of NVIDIA NVDEC (through VideoProcessingFramework), PyAV, and OpenCV (without hardware acceleration) on video decoding operation. The following metrics are observed:
1. Frame Processing Time (the time it takes to decode 1 frame)
2. CPU Utilization Across All Cores
3. Memory Utilization
4. GPU Utilization
5. GPU Memory Utilization

## Prerequisites
- Machine with [supported NVIDIA GPU](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new#Encoder)
- NVIDIA Driver
- Docker
- Git LFS

## Running the Benchmark
1. Make sure you got all the prerequisites on your system (see section above)
2. Clone the repo
3. Download [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download) and place the zip file in the root directory of this project
4. Run `git lfs pull` to download the sample videos from GitHub's Git Large File Storage
5. Build the docker image
   ```bash
   docker build -t videc-benchmark .
   ```
   2 build args are available:
   - `PROJECT_PATH`, to control where in the image you want this project to be copied to, this arg defaults to `/videc-benchmark`.
   - `VIDEO_CODEC_SDK_VERSION`, to specify which version of NVIDIA Video Codec SDK that you've downloaded from step 3, this arg defaults to 11.1.5.
6. Run the docker image:
   ```bash
   docker run --gpus all -it --pid host videc-benchmark bash
   ```
7. Enter the following command inside the container's terminal:
   ```bash
   scripts/run-benchmark.sh -i ./videos/45-seconds.mp4 -w $x
   ```
   Notes:
   - You should replace `$x` with an integer. The program will use
    the first `$x` frames as warmup during the benchmark, meaning it won't collect any data during the processing of the first `$x` frames.
   - The `./videos/45-seconds.mp4` part is the path to input file. You can look into
    the `videos` directory to see files that are available to be used as input.
8. Copy the benchmark result from the container to the host:
   ```bash
   docker cp <container ID>:/videc-benchmark/benchmark-results ./<directory name>
   ```

## Existing Report
I've done the benchmark previously and I've compiled a report document. Click [here](https://docs.google.com/document/d/1pbxKudDUY9edn-ZXURNedphEU2FucvEo6VvWe6gL-XA/edit?usp=share_link) if you need to read it.
