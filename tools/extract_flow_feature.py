import argparse
import os
import sys
import math
import glob
import cv2
import h5py
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument("h5_path", type=str, help="file path to store the extracted features")
parser.add_argument('--net_proto', type=str, 
                    default="models/hmdb51/tsn_bn_inception_flow_deploy.prototxt")
parser.add_argument('--net_weights', type=str, 
                    default="models/hmdb51_split_1_tsn_flow_reference_bn_inception.caffemodel")
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
parser.add_argument("--caffe_path", type=str, default='./lib/caffe-action/', help='path to the caffe toolbox')
args = parser.parse_args()
print args

sys.path.append('.')
sys.path.append(os.path.join(args.caffe_path, 'python'))
from pyActionRecog.action_caffe import CaffeNet


def build_vid_list():
    fpath = args.frame_path
    vid_names = [(name, os.path.join(fpath, name)) for name in os.listdir(fpath) 
                  if os.path.isdir(os.path.join(fpath, name))]
    vid_lengths = [len(glob.glob(os.path.join(vid[1],args.flow_x_prefix + "*"))) for vid in vid_names]
    vid_lengths_y = [len(glob.glob(os.path.join(vid[1],args.flow_y_prefix + "*"))) for vid in vid_names]
    for i in range(len(vid_lengths)):
        assert vid_lengths[i] == vid_lengths_y[i]
    vid = zip(*vid_names)
    vid.append(tuple(vid_lengths))
    return zip(*vid)


video_info_list = build_vid_list()

# feature_name = 'global_pool'
feature_name = "fc-action"

print(len(video_info_list))

def build_net():
    global net
    net = CaffeNet(args.net_proto, args.net_weights, 0)


def extract_single_video(video):
    global net
    video_name = video[0]
    video_frame_path = video[1]
    frame_cnt = video[2]

    stack_depth = 5

    # a small bug here that causes the mismatch between flow / imagenet
    # frame_ticks = range(1, frame_cnt, stack_depth)  # the images are 1-indexed
    frame_ticks = range(1, frame_cnt + 1, stack_depth)  # the images are 1-indexed

    frame_features = []
    for tick in frame_ticks:
        frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
        flow_stack = []
        for idx in frame_idx:
            x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
            y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
            flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
            flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
        feat = net.extract_single_flow_stack(flow_stack, feature_name, frame_size=(340, 256))
        frame_features.append(feat)

    sys.stdin.flush()
    return np.squeeze(np.array(frame_features))


build_net()

with h5py.File(args.h5_path, "a") as h5_f:
    h5_f.attrs["desc"] = "Flow feature extracted from TSN, Nx1024, each 1024 is computed from 5 flow image pairs"
    exist_keys = h5_f.keys()
    for i in tqdm(range(len(video_info_list))):
        cur_key = video_info_list[i][0]
        if cur_key not in exist_keys:
            cur_vid_feature = extract_single_video(video_info_list[i])
            h5_f.create_dataset(cur_key, data=cur_vid_feature)

