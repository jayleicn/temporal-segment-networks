#!/usr/bin/env bash

SRC_FOLDER=$1
OUT_FOLDER=$2
NUM_WORKER=$3
STEP_SIZE=$4

# original --new_width 340 --new_height 256
echo "Extracting optical flow from videos in folder: ${SRC_FOLDER}"
python tools/build_of.py ${SRC_FOLDER} ${OUT_FOLDER} --num_worker ${NUM_WORKER} --step ${STEP_SIZE} --new_width 400 --new_height 300 2>local/errors.log
