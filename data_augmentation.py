import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm # progress bar
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    x_train = sorted(glob(os.path.join(path, 'training', 'images', '*.tif')))
    y_train = sorted(glob(os.path.join(path, 'training', '1st_manual', '*.gif')))

    x_test = sorted(glob(os.path.join(path, 'test', 'images', '*.tif')))
    y_test = sorted(glob(os.path.join(path, 'test', '1st_manual', '*.gif')))

    return (x_train, y_train), (x_test, y_test)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):

        # extracting name
        name = os.path.split(x)[-1].split('.')[0]
        print(name)
        
        # read image and mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]
        # print(x.shape, y.shape)

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
        else:
            X = [x]
            Y = [y]

        
        index = 0
        for image, mask in zip (X, Y): 
            image = cv2.resize(image, size)
            mask = cv2.resize(mask, size)

            tmp_image_name = f'{name}_{index}.png'
            tmp_mask_name = f'{name}_{index}.png'

            image_path = os.path.join(save_path, 'image', tmp_image_name)
            mask_path = os.path.join(save_path, 'mask', tmp_mask_name)

            cv2.imwrite(image_path, image)
            cv2.imwrite(mask_path, mask)

            index +=1

       

if __name__ == '__main__':

    # seeding
    np.random.seed(42)

    # load data
    data_path = 'dataset'
    (x_train, y_train), (x_test, y_test) = load_data(data_path)

    # show number of train and test
    print(f'Number of train images: {len(x_train)}')
    print(f'Number of test images: {len(x_test)}')

    # create folder to save the augmented data
    create_dir('new_data/train/image/')
    create_dir('new_data/train/mask/')
    create_dir('new_data/test/image/')
    create_dir('new_data/test/mask/')

    # data augmentation
    augment_data(x_train, y_train, 'new_data/train/', augment=True)
    augment_data(x_test, y_test, 'new_data/test/', augment=False)