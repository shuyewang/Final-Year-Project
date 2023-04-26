import glob
import cv2
import numpy as np
import time
from PIL import Image
import algo.edi as edi
import multiprocessing
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

scale = [2, 4, 8]

img_dir = r"/root/autodl-tmp/DL/DATASET/DIV2K_valid_HR"
save_dir = r"/root/autodl-tmp/DL/SCALE/DIV2K_valid_HR"

cubic_save_dir = os.path.join(save_dir, "cubic")
nearest_save_dir = os.path.join(save_dir, "nearest")
lanczos4_save_dir = os.path.join(save_dir, "lanczos4")
linear_save_dir = os.path.join(save_dir, "linear")
edi_save_dir = os.path.join(save_dir, "edi")


def worker(img_file) -> None:
    file_dir, file_name = os.path.split(img_file)
    file_name, file_ext = os.path.splitext(file_name)
    if not (file_ext == ".png" or file_ext == ".jpg" or file_ext == ".bmp"):
        return
    img = cv2.imread(img_file)
    img_h, img_w, _ = img.shape
    for s in scale:
        img_lr = cv2.resize(img, dsize=(int(img_w/s), int(img_h/s)),
                            interpolation=cv2.INTER_CUBIC)

        img_neares = cv2.resize(img_lr, dsize=(
            img_w, img_h), interpolation=cv2.INTER_NEAREST)
        f = os.path.join(nearest_save_dir,
                         "{}_x{}.bmp".format(file_name, s))
        cv2.imwrite(f, img_neares)
        print("{}".format(f))

        img_lanczos4 = cv2.resize(img_lr, dsize=(
            img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
        f = os.path.join(lanczos4_save_dir,
                         "{}_x{}.bmp".format(file_name, s))
        cv2.imwrite(f, img_lanczos4)
        print("{}".format(f))

        img_linear = cv2.resize(img_lr, dsize=(
            img_w, img_h), interpolation=cv2.INTER_LINEAR)
        f = os.path.join(linear_save_dir,
                         "{}_x{}.bmp".format(file_name, s))
        cv2.imwrite(f, img_linear)
        print("{}".format(f))

        img_cubic = cv2.resize(img_lr, dsize=(
            img_w, img_h), interpolation=cv2.INTER_CUBIC)
        f = os.path.join(cubic_save_dir,
                         "{}_x{}.bmp".format(file_name, s))
        cv2.imwrite(f, img_cubic)
        print("{}".format(f))

        lr_ycrcb = cv2.cvtColor(img_lr,cv2.COLOR_BGR2YCrCb)
        lr_y, lr_cr, lr_cb = cv2.split(lr_ycrcb)
        edi_y = edi.EDI_predict(lr_y, 4, s)
        img_edi = np.zeros((edi_y.shape[0], edi_y.shape[1], 3), np.uint8)
        img_edi[:,:,0] = edi_y
        img_edi[:,:,1] = cv2.resize(lr_cr, dsize=(edi_y.shape[1], edi_y.shape[0]), interpolation=cv2.INTER_CUBIC)
        img_edi[:,:,2] = cv2.resize(lr_cb, dsize=(edi_y.shape[1], edi_y.shape[0]), interpolation=cv2.INTER_CUBIC)
        img_edi = cv2.cvtColor(img_edi, cv2.COLOR_YCrCb2BGR)

        # img_lr_b, img_lr_g, img_lr_r = cv2.split(img_lr)
        # img_edi_b = edi.EDI_predict(img_lr_b, 8, s)
        # img_edi_g = edi.EDI_predict(img_lr_g, 8, s)
        # img_edi_r = edi.EDI_predict(img_lr_r, 8, s)
        # img_edi = cv2.merge([img_edi_b, img_edi_g, img_edi_r])
        if img.shape != img_edi.shape:
                img_edi = cv2.resize(img_edi, dsize=(
                    img_w, img_h), interpolation=cv2.INTER_CUBIC)
        f = os.path.join(edi_save_dir,
                         "{}_x{}.bmp".format(file_name, s))
        cv2.imwrite(f, img_edi)
        print("{}".format(f))


if __name__ == "__main__":

    if not os.path.exists(cubic_save_dir):
        os.makedirs(cubic_save_dir)
    if not os.path.exists(nearest_save_dir):
        os.makedirs(nearest_save_dir)
    if not os.path.exists(lanczos4_save_dir):
        os.makedirs(lanczos4_save_dir)
    if not os.path.exists(linear_save_dir):
        os.makedirs(linear_save_dir)
    if not os.path.exists(edi_save_dir):
        os.makedirs(edi_save_dir)

    # img_files = os.listdir(img_dir)
    img_files = glob.glob(r"{}/*.png".format(img_dir))

    worker_pool = multiprocessing.Pool(12)
    for img_file in img_files:
        img_file = os.path.join(img_dir, img_file)
        worker_pool.apply_async(worker, args=(img_file,))
        # worker(img_file)
    worker_pool.close()
    worker_pool.join()
