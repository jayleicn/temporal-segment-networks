import sys


import caffe
from caffe.io import oversample
import numpy as np
from utils.io import flow_stack_oversample, fast_list2arr, image_array_center_crop
import cv2


class CaffeNet(object):

    def __init__(self, net_proto, net_weights, device_id, input_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        self._net = caffe.Net(net_proto, net_weights, caffe.TEST)

        input_shape = self._net.blobs['data'].data.shape

        if input_size is not None:
            input_shape = input_shape[:2] + input_size

        transformer = caffe.io.Transformer({'data': input_shape})

        if self._net.blobs['data'].data.shape[1] == 3:
            transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
            transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        else:
            pass # non RGB data need not use transformer

        self._transformer = transformer

        self._sample_shape = self._net.blobs['data'].data.shape

    def predict_single_frame(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None):

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]

        if over_sample:
            if multiscale is None:
                os_frame = oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)
        data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_flow_stack(self, frame, score_name, over_sample=True, frame_size=None):

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size) for x in frame])
        else:
            frame = fast_list2arr(frame)

        if over_sample:
            # (4 corner + 1 center) * 2
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()


    def extract_batch_rgb(self, frame, feature_layers, frame_size=None):
        """
        batch(>=1) * (3, H, W) RGB images as inputs, center crop is used.
        fc_action layer  dim=num_cls
        top_cls_global_pool dim=2048 [only for Inception V3]
        global_pool dim=1024 [only for BNInception]

        param:: frame: a list of cv2 arrays (numpy)
        param:: feature_layers: a list of blobs you want to get
        param:: frame_size: if not None, do resize
        return:: a list of data from blobs specified by feature layer
        """
        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]

        frame = fast_list2arr(frame)
  
        crop_frame = image_array_center_crop(frame, (self._sample_shape[2], self._sample_shape[3]))
        data = fast_list2arr([self._transformer.preprocess('data', x) for x in crop_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=feature_layers, data=data)
        return [out[feat_layer].copy().squeeze() for feat_layer in feature_layers]


    def extract_batch_flow_stack(self, frame, feature_layers, frame_size=None):
        """
        batch(>=1) * 5 pairs of flow images as inputs, center crop is used.
        fc_action layer  dim=num_cls
        top_cls_global_pool dim=2048 [only for Inception V3]
        global_pool dim=1024 [only for BNInception]

        param:: frame: a list of cv2 arrays (numpy)
        param:: feature_layers: a list of blobs you want to get
        param:: frame_size: if not None, do resize
        return:: a list of data from blobs specified by feature layer
        """

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size) for x in frame])
        else:
            frame = fast_list2arr(frame)
        crop_frame = image_array_center_crop(frame, (self._sample_shape[2], self._sample_shape[3]))
        crop_frame = crop_frame.reshape((-1, )+ self._sample_shape[1:])  # make batch, only needed for flow
        data = crop_frame - np.float32(128.0)
        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=feature_layers, data=data)
        return [out[feat_layer].copy().squeeze() for feat_layer in feature_layers]


