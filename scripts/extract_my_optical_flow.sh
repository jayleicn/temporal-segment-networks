#!/usr/bin/env bash

SRC_FOLDER=$1
OUT_FOLDER=$2
FPS=$3
NUM_GPU=$4

# original --new_width 340 --new_height 256
echo "Extracting optical flow from videos in folder: ${SRC_FOLDER}"
python tools/build_my_of.py ${SRC_FOLDER} ${OUT_FOLDER} --new_width 400 --new_height 300 --fps ${FPS} --num_gpu ${NUM_GPU} 2>local/errors.log
