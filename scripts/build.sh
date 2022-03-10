# Make sure you already set the PATH_TO_SDK, PATH_TO_FFMPEG, and CUDACXX environment variable
# PATH_TO_SDK > path to NVIDIA Video Codec SDK
# PATH_TO_FFMPEG > path to FFMPEG's build folder
# CUDACXX > path to CUDA Toolkit's nvcc binary

# Now the build itself
export INSTALL_PREFIX=$(pwd)/install
mkdir -p install
mkdir -p build
cd build

cmake .. \
  -DFFMPEG_DIR:PATH="$PATH_TO_FFMPEG" \
  -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL_PREFIX"

make && make install
