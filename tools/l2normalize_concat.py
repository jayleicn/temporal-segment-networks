import h5py
import os
import numpy as np
from tqdm import tqdm


def l2_normalize_numpy_array(arr, eps=1e-12):
    """Normalize numpy array (N, D) in D dim"""
    assert len(arr.shape) == 2
    norm = np.sqrt(np.sum(arr ** 2, axis=1, keepdims=True))
    norm = np.maximum(norm, eps)
    arr = arr / norm
    return arr


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.
    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.
    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def concat_normalize_h5(src_h5_file_1, src_h5_file_2, normalized_h5_file, normalize=False):
    """first normalized respectively, then concat together"""
    src_h5_1 = h5py.File(src_h5_file_1, "r")
    src_h5_2 = h5py.File(src_h5_file_2, "r")
    # src_h5_3 = h5py.File(src_h5_file_3, "r")
    normalized_h5 = h5py.File(normalized_h5_file, "a")
    keys = src_h5_1.keys()
    for i in tqdm(range(len(keys))):
        key = keys[i]
        cur_data_1 = src_h5_1[key][:]
        cur_data_2 = src_h5_2[key][:]
        # cur_data_3 = src_h5_3[key][:]
        if normalize:
            cur_data_1 = l2_normalize_numpy_array(cur_data_1)
            cur_data_2 = l2_normalize_numpy_array(cur_data_2)
            # cur_data_3 = l2_normalize_numpy_array(cur_data_3)
        cur_data = np.concatenate((cur_data_1, cur_data_2), axis=1)
        normalized_h5.create_dataset(key, data=cur_data)

    src_h5_1.close()
    src_h5_2.close()
    # src_h5_3.close()
    normalized_h5.close()

if __name__ == "__main__":
    BASE_PATH = "/net/bvisionserver4/playpen10/jielei/data/dense_flow_features"
    # flow_h5_file = os.path.join(BASE_PATH, "bfgmch_flow_hmdb51_compressed_scores.h5")
    # imagenet_h5_file = os.path.join(BASE_PATH, "bfgmch_imagenet_compressed_scores.h5")
    # places_h5_file = os.path.join(BASE_PATH, "bfgmch_places_compressed_scores.h5")
    people_file = os.path.join(BASE_PATH, "less5_people_presence.h5")
    obj_file = os.path.join(BASE_PATH, "mrcn_less5_coco80.h5")
    normalized_h5_file = os.path.join(BASE_PATH, "less5_people_mrcn_obj.h5")

    concat_normalize_h5(people_file, obj_file, normalized_h5_file, normalize=False)
