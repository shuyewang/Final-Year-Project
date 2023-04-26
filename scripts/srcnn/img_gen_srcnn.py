import glob
import multiprocessing
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

scale = [2, 4, 8]
cuda = True

input_imgs = glob.glob(r"dataset/TheDuobaoTowerStele/*.jpg")
output_dir = r"SCALE_AI/TheDuobaoTowerStele"

srcnn_save_dir = os.path.join(output_dir, "srcnn")

def model_srcnn(model, img_path, s):
    file_dir, file_name = os.path.split(img_path)
    file_name, file_ext = os.path.splitext(file_name)
    img = Image.open(img_path).convert('YCbCr')

    _tmp = img.resize((int(img.size[0]/s), int(img.size[1]/s)), Image.Resampling.BICUBIC)
    img_lr = _tmp.resize((img.size[0], img.size[1]), Image.Resampling.BICUBIC)

    y, cb, cr = img_lr.split()

    img_to_tensor = transforms.ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    if cuda:
        input = input.cuda()

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    img_hr = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')

    img_hr.save(os.path.join(srcnn_save_dir, "{}_x{}.bmp".format(file_name, s)))
    print("{}_x{}.bmp".format(file_name, s))

if __name__ == '__main__':
    if not os.path.exists(srcnn_save_dir):
        os.makedirs(srcnn_save_dir)

    for s in scale:
        device = torch.device("cuda:0" if (torch.cuda.is_available() and cuda) else "cpu")
        model_file = r'models/srcnn_x{}/model_199.pth'.format(s)
        model = torch.load(model_file, map_location=lambda storage, loc: storage).to(device)
        if cuda:
            model = model.cuda()
        else:
            model = model.cpu()
        
        worker_pool = multiprocessing.Pool(14)
        for input_img in input_imgs:
            if cuda:            
                model_srcnn(model, input_img, s)
            else:
                worker_pool.apply_async(model_srcnn, args=(model, input_img, s))
        worker_pool.close()
        worker_pool.join()
