
import glob
import json
import multiprocessing
import os
from pathlib import Path
import cv2
import numpy as np
import math
import csv
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

import algo.fsim as FSIM

scale = [2, 4, 8]

def worker(hr_img, lr_imgs_dir):
    FSIM_loss = FSIM.FSIMc()
    result = {"mse": {}, "psnr": {}, "ssim": {}, "fsim": {}}
    file_dir, file_name = os.path.split(hr_img)
    file_name, file_ext = os.path.splitext(file_name)
    hr_img_ = cv2.imread(hr_img)
    hr_img_ = cv2.cvtColor(hr_img_, cv2.COLOR_BGR2RGB)
    for s in scale:
        for algo in result.keys():
            lr_img = os.path.join(lr_imgs_dir,
                                  "{}_x{}.bmp".format(file_name, s))
            lr_img_ = cv2.imread(lr_img)
            lr_img_ = cv2.cvtColor(lr_img_, cv2.COLOR_BGR2RGB)
            if algo == "mse":
                source = mean_squared_error(hr_img_, lr_img_)
            elif algo == "psnr":
                source = peak_signal_noise_ratio(hr_img_, lr_img_)
            elif algo == "ssim":
                source = structural_similarity(
                    hr_img_, lr_img_, channel_axis=2)
            elif algo == "fsim":
                hr_ = torch.from_numpy(np.asarray(hr_img_)).permute(
                    2, 0, 1).unsqueeze(0).type(torch.FloatTensor)
                lr_ = torch.from_numpy(np.asarray(lr_img_)).permute(
                    2, 0, 1).unsqueeze(0).type(torch.FloatTensor)
                source = float(FSIM_loss(hr_, lr_))
            else:
                continue
            result[algo][s] = (lr_img, source)
            print("{}: {:.4f} - {}".format(algo, source, lr_img))
    return (hr_img, result)


if __name__ == "__main__":
    types = ('*.png', '*.jpg', '*.bmp')  # the tuple of file types

    input_imgs_dir_list = [
        r"SCALE/DIV2K_valid_HR",
        r"SCALE/Manga109",
        r"SCALE/TheDuobaoTowerStele",
        r"SCALE_AI/DIV2K_valid_HR",
        r"SCALE_AI/Manga109",
        r"SCALE_AI/TheDuobaoTowerStele"
    ]

    for input_imgs_dir in input_imgs_dir_list:
        dataset_dir = os.path.join(
            r"dataset", os.path.basename(input_imgs_dir))
        result_dir = os.path.join(
            r"RESULT", os.path.basename(input_imgs_dir))

        hr_imgs = []
        for files in types:
            hr_imgs.extend(glob.glob(os.path.join(dataset_dir, files)))

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for file_path, dir_list, file_list in os.walk(input_imgs_dir):
            for dir_name in dir_list:
                lr_imgs_dir = os.path.join(file_path, dir_name)
                # if dir_name != "srgan":
                #     continue

                worker_pool = multiprocessing.Pool(20)
                result_feature = []
                for hr_img in hr_imgs:
                    result_feature.append(worker_pool.apply_async(
                        worker, args=(hr_img, lr_imgs_dir)))
                worker_pool.close()
                worker_pool.join()

                result_list = {}
                for r in result_feature:
                    tmp = r.get()
                    result_list[tmp[0]] = tmp[1]

                result_file = os.path.join(
                    result_dir, "{}.json".format(dir_name))
                with open(result_file, 'w') as f:
                    f.write(json.dumps(result_list))
                    print("write {}".format(result_file))

                # for algo in result.keys():
                #     result_csv = os.path.join(
                #         result_out_dir, "{}_{}.csv".format(algo, dir_name))
                #     with open(result_csv, 'w') as f:
                #         print("write {}".format(result_csv))
                #         w = csv.writer(f)
                #         w.writerow(result[algo].keys())
                #         w.writerows(zip(*result[algo].values()))
