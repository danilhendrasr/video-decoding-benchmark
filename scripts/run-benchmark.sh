#!/usr/bin/bash

while getopts ":bv" flags; do
  case "${flags}" in
  b)
    $(pwd)/scripts/build.sh
    ;;
  v)
    $(pwd)/scripts/fetch-videos.sh
    ;;
  esac
done

mkdir -p $(pwd)/benchmark-results/plot/fpt
mkdir -p $(pwd)/benchmark-results/plot/cpu
mkdir -p $(pwd)/benchmark-results/plot/mem
mkdir -p $(pwd)/benchmark-results/plot/gpu
mkdir -p $(pwd)/benchmark-results/plot/gpu-mem
mkdir -p $(pwd)/benchmark-results/csv/fpt
mkdir -p $(pwd)/benchmark-results/csv/cpu
mkdir -p $(pwd)/benchmark-results/csv/mem
mkdir -p $(pwd)/benchmark-results/csv/gpu
mkdir -p $(pwd)/benchmark-results/csv/gpu-mem
python3 ./install/bin/main.py 0 ./videos/45-seconds.mkv ./benchmark-results/out.native
