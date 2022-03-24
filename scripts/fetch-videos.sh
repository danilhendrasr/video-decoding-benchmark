if [ -f $(pwd)/videos/5-minutes.mkv ]; then
  echo "Video sample already exists, skip fetching video"
else
  ffmpeg -t 300 -i https://cctvjss.jogjakota.go.id/rthp/rthp_segoro_amarto_tegalrejo_2.stream/playlist.m3u8 -c copy $(pwd)/videos/5-minutes.mkv
fi
