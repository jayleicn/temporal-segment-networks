import os
import glob
import sys
import cv2
from pipes import quote
from multiprocessing import Pool, current_process

from tqdm import tqdm
import numpy as np
import argparse

sys.path.append("lib/caffe-action/python/")
sys.path.append("lib/dense_flow/build")
from pyActionRecog.action_flow import FlowExtractor


def single_video_rgb_flow_extraction(vid_path, dev_id=0):
    """
    extract optical flow and rgb images from a video with 15 flow image
    pairs per second. (that means 18 rgn images per second)
    extraction indices:
    fps=30: [1-6], [11-16], [21-26] + 30*n
    fps=24: [1-6], [9-14], [17-22] + 24*n
    """
    vidcap = cv2.VideoCapture(vid_path)
    num_frm = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    max_seconds = int(num_frm / FPS) + 1
    frm_indices = []
    if FPS == 30:
        for i in range(max_seconds):
            frm_indices.extend(np.arange(1,7) + i*30)
            frm_indices.extend(np.arange(11,17) + i*30)
            frm_indices.extend(np.arange(21,27) + i*30)
    elif FPS == 24:
        for i in range(max_seconds):
            frm_indices.extend(np.arange(1,7) + i*24)
            frm_indices.extend(np.arange(9,15) + i*24)
            frm_indices.extend(np.arange(17,23) + i*24) 
            
    # extract RGB frames
    success, image = vidcap.read()
    cnt = 1
    success = True
    frm_stack = []
    while success:
        if cnt in frm_indices:
            if NEW_SIZE is not None:
                try:
                    frm_stack.append(cv2.resize(image, NEW_SIZE))
                except cv2.error as e:
                    print("The error video path %s" % vid_path)
                    print("Frame No. %d" % cnt)
            else:
                frm_stack.append(image)      
#         cv2.imwrite(frame_save_path_pattern % count, cv2.resize(image, NEW_SIZE))     # save frame as JPEG file
        success,image = vidcap.read()
        # print 'Read a new frame: ', success
        cnt += 1
    
    # extract flow
    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    flow_extractor = FlowExtractor(dev_id) 

    flow_stack = []
    frm_stack_to_save = []
    num_groups = len(frm_indices) / 6 + 1
    for i in range(num_groups):
        cur_6_frm = frm_stack[i*6:(i+1)*6] # 6 rgb images
        if len(cur_6_frm) > 1:
            flow_stack.append(flow_extractor.extract_flow(cur_6_frm)) # 10 flow images / 5 pairs
            frm_stack_to_save.extend(cur_6_frm[:-1])
    flow_stack = np.concatenate(flow_stack)
    num_frm_to_save = len(frm_stack_to_save)
    assert(len(flow_stack) == 2 * len(frm_stack_to_save))

    # save all images
    save_path = os.path.join(OUT_PATH, os.path.basename(vid_path).split(".")[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rgb_pattern = os.path.join(save_path, "img_%05d.jpg")
    x_flow_patten = os.path.join(save_path, "flow_x_%05d.jpg")
    y_flow_patten = os.path.join(save_path, "flow_y_%05d.jpg")
    for i in range(num_frm_to_save):
        cv2.imwrite(rgb_pattern % (i+1), frm_stack_to_save[i])
        cv2.imwrite(x_flow_patten % (i+1), flow_stack[2*i])
        cv2.imwrite(y_flow_patten % (i+1), flow_stack[2*i+1])
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--ext", type=str, default='mp4', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--num_gpu", type=int, default=4, help='gpus to use')
    parser.add_argument("--fps", type=int, default=30, choices=[24,30], help='video frame rate')
    parser.add_argument("--new_width", type=int, default=400, help='resize image width')
    parser.add_argument("--new_height", type=int, default=300, help='resize image height')

    args = parser.parse_args()

    src_path = args.src_dir
    OUT_PATH = args.out_dir
    ext = args.ext
    FPS = args.fps
    NEW_SIZE = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu

    # src_path = "scripts/extract_my_optical_flow.sh /net/bvisionserver4/playpen1/jielei/data/videos/bbt/bbt_clips"
    # OUT_PATH = "/net/bvisionserver4/playpen1/jielei/data/preprocessed_video_data/dense_flow_frames_step1_new/bbt"
    # ext = "mp4"
    # FPS = "24"
    # NEW_SIZE = (400, 300)
    # NUM_GPU = 
    if not os.path.isdir(OUT_PATH):
        print "creating folder: "+OUT_PATH
        os.makedirs(OUT_PATH)

    file_pattern = os.path.join(src_path, '*/*.'+ext)
    vid_list = glob.glob(file_pattern)
    exists_file_list = [name for name in os.listdir(OUT_PATH) if os.path.isdir(os.path.join(OUT_PATH, name))]
    vid_list = [x for x in vid_list if os.path.basename(x).split(".")[0] not in exists_file_list]

    print("Number of videos to extract: %d" % len(vid_list))
    pool = Pool(NUM_GPU*2)
    r = list(tqdm(pool.imap(single_video_rgb_flow_extraction, vid_list), total=len(vid_list)))

