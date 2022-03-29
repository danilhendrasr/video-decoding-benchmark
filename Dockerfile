FROM nvcr.io/nvidia/cuda:11.2.2-devel-ubuntu18.04
ARG PROJECT_PATH=/videc-benchmark
ARG VIDEO_CODEC_SDK_VERSION=11.1.5

ENV VIDEOSDK_VERSION ${VIDEO_CODEC_SDK_VERSION}
ENV CUDA_PKG_VERSION 11-2
LABEL com.nvidia.videosdk.version="${VIDEOSDK_VERSION}"

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},video
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  software-properties-common \
  cuda-cudart-dev-${CUDA_PKG_VERSION} \
  cuda-driver-dev-${CUDA_PKG_VERSION} && \
  add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
  apt-get install -y --no-install-recommends ffmpeg unzip curl cmake \
  pkg-config python3.8 python3-pip python3-setuptools python3.8-dev \
  python-dev libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libswscale-dev libswresample-dev libavfilter-dev && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \ 
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
  update-alternatives --config python3

WORKDIR $PROJECT_PATH
COPY requirements.txt .

RUN pip3 install --upgrade pip && \
  pip3 install --upgrade cython && \
  pip3 install -r requirements.txt

WORKDIR /opt/nvidia/video-sdk
COPY Video_Codec_SDK_${VIDEOSDK_VERSION}.zip /opt/nvidia/video-sdk

RUN unzip -j Video_Codec_SDK_${VIDEOSDK_VERSION}.zip \
  Video_Codec_SDK_${VIDEOSDK_VERSION}/Interface/cuviddec.h \
  Video_Codec_SDK_${VIDEOSDK_VERSION}/Interface/nvcuvid.h \
  Video_Codec_SDK_${VIDEOSDK_VERSION}/Interface/nvEncodeAPI.h \
  -d /usr/local/cuda/include && \
  unzip -j Video_Codec_SDK_${VIDEOSDK_VERSION}.zip \
  Video_Codec_SDK_${VIDEOSDK_VERSION}/Lib/linux/stubs/x86_64/libnvcuvid.so \
  Video_Codec_SDK_${VIDEOSDK_VERSION}/Lib/linux/stubs/x86_64/libnvidia-encode.so \
  -d /usr/local/cuda/lib64/stubs && \
  unzip Video_Codec_SDK_${VIDEOSDK_VERSION}.zip && rm Video_Codec_SDK_${VIDEOSDK_VERSION}.zip

WORKDIR $PROJECT_PATH
COPY . .

ENV PYTHONPATH=$PROJECT_PATH/install/bin:$PYTHONPATH

RUN bash -c 'mkdir -p benchmark-results/{plot,csv}/{fpt,cpu,mem,gpu,gpu_mem}' && \
  bash -c 'mkdir -p {install,build}' && cd build && \
  export PATH_TO_SDK=/opt/nvidia/video-sdk/Video_Codec_SDK_${VIDEOSDK_VERSION} && \
  cmake .. \
  -DFFMPEG_DIR:PATH="/usr/bin" \
  -DFFMPEG_INCLUDE_DIR:PATH="/usr/include/x86_64-linux-gnu" \
  -DFFMPEG_LIB_DIR:PATH="/usr/lib/x86_64-linux-gnu" \
  -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DCMAKE_INSTALL_PREFIX:PATH="$PROJECT_PATH/install" \
  -DPYTHON_EXECUTABLE:FILEPATH="/usr/bin/python3.8" \
  -DPYTHON_INCLUDE_DIR:PATH="/usr/include/python3.8"  \
  -DPYTHON_LIBRARY:FILEPATH="/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so" \
  --trace-source=CMakeLists.txt --trace-expand && \
  make && make install