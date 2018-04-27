bash scripts/extract_optical_flow.sh \
     /net/bvisionserver4/playpen10/jielei/notebook/tqa_baseline/data/test_flow_extraction/test_videos \
     /net/bvisionserver4/playpen10/jielei/notebook/tqa_baseline/data/test_flow_extraction/test_frames 2 10

./build/extract_gpu \
    -f friends_s03e06_seg02_clip_19.mp4 \
    -x tmp/flow_x -y tmp/flow_y -i tmp/image -b 20 -t 1 -d 0 -s 10 -o dir

./build/extract_gpu \
    -f grey_s03e25_seg02_clip_05.mp4 \
    -x grey_hug/flow_x -y grey_hug/flow_y -i grey_hug/image -b 20 -t 1 -d 0 -s 10 -o dir