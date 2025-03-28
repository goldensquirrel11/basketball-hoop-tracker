#!/usr/bin/bash

for f in *.mp4
do
    ffmpeg -i $f -filter:v scale=1920:-1 -c:a copy -start_number 0 "${f%.*}"_%06d.png
done
