#!/usr/bin/bash

for f in *.avi
do
    ffmpeg -i $f -start_number 0 "${f%.*}"_%06d.png
done
