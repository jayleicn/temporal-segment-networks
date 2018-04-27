import h5py
import os
import numpy as np
from tqdm import tqdm

def fix_h5(err_h5_file, fixed_h5_file, target_h5_file):
    """match flow dim0 and imagenet dim0"""
    err_h5 = h5py.File(err_h5_file, "r")
    fixed_h5 = h5py.File(fixed_h5_file, "a")
    tar_h5 = h5py.File(target_h5_file, "r")
    keys = err_h5.keys()
    not_in_cnt = 0
    for i in tqdm(range(len(keys))):
        key = keys[i]
        if key not in tar_h5.keys():
            not_in_cnt = not_in_cnt + 1
            continue
        if err_h5[key].shape[0] != tar_h5[key].shape[0]:
            if err_h5[key].shape[0] + 1 == tar_h5[key].shape[0]:
               cur_err_data =  err_h5[key][:]
               cur_fixed_data = np.concatenate((cur_err_data, cur_err_data[-1].reshape(1, -1)), axis=0)
            else:
                print("Not by adding 1 more dim")
                print(i, key)
                continue
        else:
            cur_fixed_data = err_h5[key][:]
        fixed_h5.create_dataset(key, data=cur_fixed_data)
   
    err_h5.close()
    fixed_h5.close()
    tar_h5.close()

if __name__ == "__main__":
    BASE_PATH = "/net/bvisionserver4/playpen10/jielei/data/dense_flow_features"
    err_path = os.path.join(BASE_PATH, "bfgmch_flow.h5")
    fix_path = os.path.join(BASE_PATH, "bfgmch_flow_fixed.h5") 
    tar_path = os.path.join(BASE_PATH, "bfgmch_imagenet.h5")

    fix_h5(err_path, fix_path, tar_path)