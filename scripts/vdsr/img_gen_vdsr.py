import glob
import time
import torch
import os
import numpy as np
from torch.autograd import Variable
import cv2

input_imgs = glob.glob(r"/root/autodl-tmp/DL/DATASET/TESTIMG/*.jpg")
output_dir = r"/root/autodl-tmp/DL/SCALE_AI/TESTIMG"

scale = [2, 4, 8]

model_file = r'/root/autodl-tmp/DL/DLL/models/vdsr248/model_epoch_5.pth'
cuda = True
gpus = 0

vdsr_save_dir = os.path.join(output_dir, "vdsr248")

def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:, :, 0] = y
    img[:, :, 1] = ycbcr[:, :, 1]
    img[:, :, 2] = ycbcr[:, :, 2]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return img


def model_vdsr(model, img_lr):
    img_lr_y = img_lr[:, :, 0].astype(float)

    input = img_lr_y/255.

    input = Variable(torch.from_numpy(input).float()).view(
        1, -1, input.shape[0], input.shape[1])

    if cuda:
        model = model.cuda()
        input = input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    out = model(input)
    elapsed_time = time.time() - start_time
    print("{}ms".format(elapsed_time))

    out = out.cpu()

    im_h_y = out.data[0].numpy().astype(np.float32)

    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.

    return colorize(im_h_y[0, :, :], img_lr)


if __name__ == '__main__':
    if not os.path.exists(vdsr_save_dir):
        os.makedirs(vdsr_save_dir)

    if cuda:
        print("=> use gpu id: '{}'".format(gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id")

    model = torch.load(model_file)
    model = model["model"]

    for input_img in input_imgs:
        file_dir, file_name = os.path.split(input_img)
        file_name, file_ext = os.path.splitext(file_name)
        img = cv2.imread(input_img)
        for s in scale:
            img_lr = cv2.resize(img, dsize=(int(img.shape[1]/s), int(img.shape[0]/s)),
                                interpolation=cv2.INTER_CUBIC)
            img_lr = cv2.resize(img_lr, dsize=(int(img.shape[1]), int(img.shape[0])),
                                interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite(os.path.join(output_dir, "{}_x{}_input.bmp".format(file_name, s)), img_lr)
            
            print("{}".format(input_img))
            img_lr_ycrcb = cv2.cvtColor(img_lr,cv2.COLOR_BGR2YCrCb)
            img_hr = model_vdsr(model, img_lr_ycrcb)
            cv2.imwrite(os.path.join(vdsr_save_dir, "{}_x{}.bmp".format(file_name, s)), img_hr)
