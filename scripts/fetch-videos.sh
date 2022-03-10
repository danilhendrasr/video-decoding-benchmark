if [ -f $(pwd)/videos/45-seconds.mkv ]; then
  echo "Video sample already exists, skip fetching video"
else
  ffmpeg -i https://cctvjss.jogjakota.go.id/rthp/rthp_segoro_amarto_tegalrejo_2.stream/playlist.m3u8 -c copy -t 45 $(pwd)/videos/45-seconds.mkv
fi
