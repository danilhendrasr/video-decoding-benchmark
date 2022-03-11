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

mkdir -p $(pwd)/benchmark-results/plot/cpu
python3 ./install/bin/main.py 0 ./videos/45-seconds.mkv ./benchmark-results/out.native
