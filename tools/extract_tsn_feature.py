import argparse
import os
import sys
import math
import glob
import cv2
import h5py
import numpy as np
from tqdm import tqdm

# suppress caffe info
os.environ['GLOG_minloglevel'] = '2'


def extract_single_video(video, feature_names):
    """
    param video is a tuple contains two elements
    every 5 consecutive flow image pairs are sent to the model 
    """
    global net
    video_name = video[0]
    video_frame_path = video[1]
    frame_cnt = video[2]

    if args.modality == "flow":
        stack_depth = 5
    elif args.modality == "rgb":
        stack_depth = 1

    # a small bug here that causes the mismatch between flow / imagenet
    # frame_ticks = range(1, frame_cnt, stack_depth)  # the images are 1-indexed
    frame_ticks = range(1, frame_cnt + 1, stack_depth)  # the images are 1-indexed

    # extract every 100 stack ???
    frame_features = [] 
    frame_scores = []
    cnt = 0
    img_stack = []
    for tick in frame_ticks:
        cnt = cnt +1
        if args.modality == "flow":
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
            for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
                img_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                img_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
        elif args.modality == "rgb":
            name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
            img_stack.append(cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR))

        if cnt == 100 or tick == frame_ticks[-1]:
            if args.modality == "flow":      
                cur_feature, cur_score = net.extract_batch_flow_stack(img_stack, feature_names, frame_size=None)
            elif args.modality == "rgb":      
                cur_feature, cur_score = net.extract_batch_rgb(img_stack, feature_names, frame_size=(400, 300))

            if len(cur_feature.shape) == 1:
                cur_feature = np.expand_dims(cur_feature, axis=0)
                cur_score = np.expand_dims(cur_score, axis=0)
            frame_features.append(cur_feature)
            frame_scores.append(cur_score)
            cnt = 0
            img_stack = []

    sys.stdin.flush()
    if len(frame_features) == 1:  # less than 100 RGB images or 500 flow image paris
        return frame_features[0], frame_scores[0]
    else:
        return np.concatenate(frame_features, axis=0), np.concatenate(frame_scores, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_path', type=str, help="root directory holding the frames")
    parser.add_argument("--h5_path_feat", type=str, help="file path to store the extracted features")
    parser.add_argument("--h5_path_score", type=str, help="file path to store the extracted features")
    parser.add_argument("--modality", type=str, help="flow or rgb")
    parser.add_argument('--net_proto', type=str)
    parser.add_argument('--net_weights', type=str)
    parser.add_argument("--rgb_prefix", type=str, default="", help="prefix of RGB images")
    parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
    parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
    parser.add_argument("--caffe_path", type=str, default='./lib/caffe-action/', help='path to the caffe toolbox')
    args = parser.parse_args()
    print(args)

    if args.modality == "flow":
        args.net_proto = "models/kinetics400/inception_v3_flow_deploy.prototxt"
        args.net_weights = "models/kinetics400/inception_v3_flow_kinetics.caffemodel"
    elif args.modality == "rgb":
        args.net_proto = "models/kinetics400/inception_v3_rgb_deploy.prototxt"
        args.net_weights = "models/kinetics400/inception_v3_kinetics_rgb_pretrained.caffemodel" 
    else:
        print("Wrong value for arg modality")
        sys.exit(1)

    sys.path.append('.')
    sys.path.append(os.path.join(args.caffe_path, 'python'))
    from pyActionRecog.action_caffe import CaffeNet

    def build_vid_list():
        fpath = args.frame_path
        vid_names = [(name, os.path.join(fpath, name)) for name in os.listdir(fpath) 
                    if os.path.isdir(os.path.join(fpath, name))]
        if args.modality == "flow":
            vid_lengths = [len(glob.glob(os.path.join(vid[1],args.flow_x_prefix + "*.jpg"))) for vid in vid_names]
            vid_lengths_y = [len(glob.glob(os.path.join(vid[1],args.flow_y_prefix + "*.jpg"))) for vid in vid_names]
            for i in range(len(vid_lengths)):
                assert vid_lengths[i] == vid_lengths_y[i]
        elif args.modality == "rgb":
            vid_lengths = [len(glob.glob(os.path.join(vid[1],args.rgb_prefix + "*.jpg"))) for vid in vid_names]
        vid = zip(*vid_names)
        vid.append(tuple(vid_lengths))
        return zip(*vid)

    video_info_list = build_vid_list()

    def build_net():
        global net
        net = CaffeNet(args.net_proto, args.net_weights, 0)

    build_net()

    # feature_name = 'global_pool'
    feature_names_to_extract = ["top_cls_global_pool", "fc_action"]

    with h5py.File(args.h5_path_feat, "a") as h5_feat:
        with h5py.File(args.h5_path_score, "a") as h5_score:
            h5_feat.attrs["desc"] = "Flow or action RGB feature extracted from TSN, Nx2048 or Nx1024, fps=3"
            h5_feat.attrs["desc"] = "Flow or action RGB feature extracted from TSN, Nx2048 or Nx1024, fps=3"
            exist_keys = h5_score.keys()
            video_info_list = [ele for ele in video_info_list if ele[0] not in exist_keys]
            print("Number of videos to process %d" % len(video_info_list))
            for i in tqdm(range(len(video_info_list))):
                cur_key = video_info_list[i][0]
                try:
                    cur_vid_feature, cur_vid_score = extract_single_video(video_info_list[i], feature_names_to_extract)
                    # print(cur_key)
                    # print(cur_vid_feature.shape)
                    # print(cur_vid_score.shape)
                    # sys.exit(0)
                    h5_feat.create_dataset(cur_key, data=cur_vid_feature)
                    h5_score.create_dataset(cur_key, data=cur_vid_score)
                except Exception as e:
                    print(e.message)
                    print(video_info_list[i])

