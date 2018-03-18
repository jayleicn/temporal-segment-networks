from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lmdb
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import pickle
import glob
import time
import sys
import os
import math

from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image


class PlacesResNet50Feature(nn.Module):
    def __init__(self):
        super(PlacesResNet50Feature, self).__init__()
        arch = 'resnet50'
        # load the pre-trained weights
        model_file = 'whole_%s_places365_python36.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        places_resnet50 = torch.load(model_file)
        self.res5c_feature = nn.Sequential(*list(places_resnet50.children())[:-1])

    def forward(self, x):
        """
        res5c -> batch x 2048
        (pool5 can be obtained by using 7x7 avg-pooling on res5c)
        """
        x = self.res5c_feature(x)
        return x


class ImageNetResNet152Feature(nn.Module):
    def __init__(self):
        super(ImageNetResNet152Feature, self).__init__()
        resnet152 = models.resnet152(pretrained=True)
        self.res5c_feature = nn.Sequential(*list(resnet152.children())[:-1])

    def forward(self, x):
        """
        pool5 -> batch x 2048, obtained by using 7x7 avg-pooling on res5c
        """
        x = self.res5c_feature(x)
        return x


def make_image_tensor(image_paths):
    tensors = []
    for ele in image_paths:
        image = Image.open(ele).convert('RGB')
        image = imagenet_transform(image)
        image = image.view(1, 3, 224, 224)
        tensors.append(image)
    return torch.cat(tensors, 0)


def get_video_feature(frame_path):
    """
    input:
        path to the frames for a single video
    return:
        image features for the frames
    """
    max_bsz = 300
    stack_length = 5
    file_paths = glob.glob(os.path.join(frame_path, "img_*"))
         
    num_frames = len(file_paths)
    frame_ticks = range(0, num_frames, max_bsz)
    feature_list = []
    for b_idx in range(len(frame_ticks)):
        cur_file_paths = file_paths[frame_ticks[b_idx] : frame_ticks[b_idx] + max_bsz]
        try:
            inputs = make_image_tensor(cur_file_paths)
        except:
            e = sys.exc_info()[0]
            print('Err in function get_video_feature: %s \n' % e)
            # continue
        inputs = Variable(inputs, volatile=True)
        inputs = inputs.cuda()
        feat = extractor(inputs)  # N x 2048
        feat = feat.data
        feature_list.append(feat)
    if len(frame_ticks) > 1:
        features = torch.cat(feature_list, dim=0)
    else:
        features = feature_list[0]

    assert len(features) == num_frames

    temporal_pooled_list = []
    frame_ticks = range(0, num_frames, stack_length)
    for tick in frame_ticks:
        start = tick
        end = min(tick+stack_length, num_frames)
        temporal_pooled_list.append(torch.mean(features[start:end], dim=0, keepdim=True))
    temporal_pooled = torch.cat(temporal_pooled_list, dim=0) 

    return temporal_pooled.squeeze().cpu().numpy()


def extract_all(args, video_names):
    """
    input:
        video_names(list) - 
        store_switch: a list of integers, selecting which features to be stored.
    return:

    """
    pool5_path = args.pool5_h5_file
    print("extract all")
    f_pool5 = h5py.File(pool5_path, "a")
    exist_keys = f_pool5.keys()
    f_pool5.attrs["desc"] = "feature from resnet152-pool5 output, size (Nx2048), image featrues \
            from the same video file, N is varied, but at most 300."
    print("Pool5 selected.")
    for i in tqdm(range(len(video_names))):
        cur_subdir = video_names[i]
        if cur_subdir not in exist_keys:
            cur_path = os.path.join(args.frame_dir, cur_subdir)
            try:
                data_pool5 = get_video_feature(cur_path)
            except Exception as e:
                print(e)
                continue
            f_pool5.create_dataset(cur_subdir, data=data_pool5)
    f_pool5.close()


if __name__ == "__main__":
    # settings
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", type=str, default="imagenet", help="imagenet / places")
    parser.add_argument("frame_dir", type=str, default=None)
    parser.add_argument("pool5_h5_file", type=str, default=None)
    args = parser.parse_args()
    print(args)

    sub_dirs = [d for d in os.listdir(args.frame_dir) if os.path.isdir(os.path.join(args.frame_dir, d))]
    # vid_lengths = [len(glob.glob(os.path.join(args.frame_dir, "img_" + "*"))) for vid in sub_dirs]
    # all_len_list = []
    # for i in tqdm(range(len(sub_dirs))):
    #     cur_files = os.listdir(os.path.join(frame_dir, sub_dirs[i]))
    #     cur_len = len(cur_files)
    #     all_len_list.append(cur_len)

    # Step 1, Define feature extractor (nn.Module)
    # https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-152-deploy.prototxt
    # see the link above for resnet architectrue, layer_name, etc.
    print("\n[Phase 1] Setup feature extractor.")
    # extractor = resnet152_feature()
    if args.feature == "imagenet":
        extractor = ImageNetResNet152Feature()
    elif args.feature == "places": 
        extractor = PlacesResNet50Feature()
    else:
        raise NotImplementedError

    # Step 2, set experiment settings
    print("\n[Phase 2] Config settings.")

    USE_CUDA = torch.cuda.is_available()
    if not USE_CUDA:
        print("no GPU available")
        sys.exit(1)
    extractor.cuda()
    cudnn.benchmark = True

    extractor.eval()  # must set to eval model

    # testing
    sample_input = Variable(torch.randn(300, 3, 224, 224), volatile=True)
    if USE_CUDA:
        sample_input = sample_input.cuda()
        print(" Extraction on GPU.")

    sample_output = extractor(sample_input)
    featureSize = sample_output.size()
    print(" Feature Size is: ", featureSize)

    imagenet_transform = transforms.Compose([
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            
    print("\n[Phase 3] : Feature Extraction")
    extract_all(args, sub_dirs)
