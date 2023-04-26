# This is the explanation of the code for Single Image Super-resolution through fusion of up-sampling methods.
## File 'TheDuobaoTowerStele'
File 'TheDuobaoTowerStele' includes 40 images used for testing different interpolation methods

## File 'models'
File 'models' contains the trained models for testing deep learning based methods.

## File 'scripts'
1. File 'algo' includes one taditional method called [New Edge-Directed Interpolation (NEDI)](https://github.com/Kirstihly/Edge-Directed_Interpolation.git), the other evaluation metric called [Feature Similarity Index (FSIMc, improved from FSIM)](https://github.com/mikhailiuk/pytorch-fsim.git).
2. File 'srcnn' has three codes; the first one 'img_gen_srcnn' is used for generating SRCNN interpolated images, the second named 'model' is the structure of SRCNN, the last called 'test' is written to generate the feature map for each layer.
3. File 'vdsr' contains three codes; 'img_gen_vdsr' is used for generating images that interpolated by VDSR, 'test' is used for produce feature map for each layer, 'vdsr' shows the structure of VDSR.
4. The code 'img_detail' is written to generate a cropped detail for image 0882 from DIV2K dataset to compare the effects between each method.
5. 'img_gen' is used for generating images that interpolated by different traditional methods.
6. The code 'img_result' is written to illustrate the overall score of different evaluation metrics on different SISR methods.
7. 'img_test' is the evaluation of four metrics on different interpolation methods.

## File 'train'

For 'pytorch-vdsr', the matlab code 'generate_train' is modified to generate various h5 files instead of one large file to prevent overflow. The 'merge' aims to merge the various h5 file to one again. The python code 'dataset' is used for reading the h5 file and modified to calculate the length of the dataset avoid opening the h5 file repeatedly, which imporve the efficiency. The parameters for training the VDSR in 'main_vdsr' is modified in terminal but not in the code; The 'step' is modified to 1 and the 'nEpochs' is modified to 5.  