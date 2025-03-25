#!/usr/bin/bash

# mkdir resized

for f in *.png
do
    ffmpeg -i $f -vf scale=1920:1080 resized/$f -y
done