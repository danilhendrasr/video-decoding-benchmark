# Make sure you already set the PATH_TO_SDK, PATH_TO_FFMPEG, and CUDACXX environment variable
# PATH_TO_SDK > path to NVIDIA Video Codec SDK
# CUDACXX > path to CUDA Toolkit's nvcc binary

export PATH_TO_FFMPEG=$(pwd)/ffmpeg_5.1_build

# Now the build itself
virtualenv videc_benchmark_env
source videc_benchmark_env/bin/activate
pip install -r requirements.txt

export CMAKE_INSTALL_PREFIX=$(pwd)/install
mkdir -p install
mkdir -p build
cd build

cmake .. \
  -DFFMPEG_DIR:PATH="$PATH_TO_FFMPEG" \
  -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DCMAKE_INSTALL_PREFIX:PATH="$CMAKE_INSTALL_PREFIX"

make && make install
