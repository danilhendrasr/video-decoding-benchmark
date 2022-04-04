default_input_file='./videos/45-seconds.mp4'
default_warmup_frames=20

warmup_frames=$default_warmup_frames
input_file=$default_input_file

while getopts ":bi:w:" flags; do
  case "${flags}" in
  b)
    $(pwd)/scripts/build.sh
    ;;
  i)
    input_file=${OPTARG}
    ;;
  w)
    warmup_frames=${OPTARG}
    ;;
  esac
done

echo "This program compares NVIDIA NVDEC, PyAV, and OpenCV performance on video decoding."
echo "Usage: run-benchmark.sh -b -i \$path_to_input_file -w \$warmup_frames_count"
echo "-b  Optional, build the project first or not, you only need to do this of you change the source code inside the container or you're running without container"
echo "-i  Optional, defaults to $default_input_file"
echo "-w  Optional, defaults to $default_warmup_frames"
echo ""

echo "Benchmarking OpenCV..."
python3 ./install/bin/main.py opencv $input_file $warmup_frames

echo "Benchmarking PyAV..."
python3 ./install/bin/main.py pyav $input_file $warmup_frames

echo "Benchmarking NVDEC..."
python3 ./install/bin/main.py nvdec $input_file $warmup_frames

python3 ./install/bin/aggregate_report.py

echo ""
echo "Benchmark finished"
echo "Benchmark result written to: $(pwd)/benchmark-results/report.md"
