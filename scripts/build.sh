# Make sure you already set the PATH_TO_SDK, PATH_TO_FFMPEG, and CUDACXX environment variable
# PATH_TO_SDK > path to NVIDIA Video Codec SDK
# CUDACXX > path to CUDA Toolkit's nvcc binary

# Now the build itself
virtualenv videc_benchmark_env
source videc_benchmark_env/bin/activate
pip install -r requirements.txt

export CMAKE_INSTALL_PREFIX="$(pwd)/install"
mkdir -p {install,build}
mkdir -p benchmark-results/{csv,plot}/{fpt,cpu,mem,gpu,gpu_mem}
mkdir benchmark-results/individual_summary
cd build

cmake .. \
  -DFFMPEG_DIR:PATH="/usr/bin" \
  -DFFMPEG_INCLUDE_DIR:PATH="/usr/include/x86_64-linux-gnu" \
  -DFFMPEG_LIB_DIR:PATH="/usr/lib/x86_64-linux-gnu" \
  -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
  -DPYTHON_EXECUTABLE:FILEPATH="/usr/bin/python3.8" \
  -DPYTHON_INCLUDE_DIR:PATH="/usr/include/python3.8" \
  -DPYTHON_LIBRARY:FILEPATH="/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DCMAKE_INSTALL_PREFIX:PATH="$CMAKE_INSTALL_PREFIX" \
  --trace-source=CMakeLists.txt --trace-expand

make && make install
