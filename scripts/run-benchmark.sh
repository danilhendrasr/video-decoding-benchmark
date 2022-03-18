#!/usr/bin/bash

warmup_iteration=0

while getopts ":bvw:" flags; do
  case "${flags}" in
  b)
    $(pwd)/scripts/build.sh
    ;;
  v)
    $(pwd)/scripts/fetch-videos.sh
    ;;
  w)
    warmup_iteration=${OPTARG}
    ;;
  esac
done

mkdir -p $(pwd)/benchmark-results/plot/fpt
mkdir -p $(pwd)/benchmark-results/plot/cpu
mkdir -p $(pwd)/benchmark-results/plot/mem
mkdir -p $(pwd)/benchmark-results/plot/gpu
mkdir -p $(pwd)/benchmark-results/plot/gpu_mem

mkdir -p $(pwd)/benchmark-results/csv/fpt
mkdir -p $(pwd)/benchmark-results/csv/cpu
mkdir -p $(pwd)/benchmark-results/csv/mem
mkdir -p $(pwd)/benchmark-results/csv/gpu
mkdir -p $(pwd)/benchmark-results/csv/gpu_mem

python3 ./install/bin/main.py ./videos/45-seconds.mkv $warmup_iteration
