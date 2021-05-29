import os
import numpy as np
import cv2 as cv
import pandas as pd 

IMG_ORIGINAL_SHAPE = (1365, 2048, 3)

def center_crop(image,out_height,out_width):
    image_height, image_width = image.shape[:2]
    offset_height = (image_height - out_height) // 2
    offset_width = (image_width - out_width) // 2
    image = image[offset_height:offset_height+out_height, offset_width:offset_width+out_width,:]
    return image

def resize_maintain_aspect(image,target_h,target_w):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        new_width = target_w
        new_height = int(image_height*(target_w/image_width))
    else:
        new_height = target_h
        new_width = int(image_width*(target_h/image_height))

    image = cv.resize(image,(new_width,new_height),interpolation=cv.INTER_CUBIC)
    return image

def npy_converter(image_path, image_height,image_width, output_path):
    # open image to numpy array
    img = cv.imread(image_path)

    # resize
    img = resize_maintain_aspect(img,image_height,image_width)

    # center crop to target height & width
    img = center_crop(img,image_height,image_width)

    # switch to RGB from BGR
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    np.save(output_path, img, allow_pickle=True)

IMAGES_PATH = 'images/'
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

def get_image_path(filename):
    return (IMAGES_PATH + filename + '.jpg')

df_train['image_path'] = df_train['image_id'].apply(get_image_path)
df_test['image_path'] = df_test['image_id'].apply(get_image_path)
train_labels = df_train.loc[:, 'healthy':'scab']
train_paths = df_train.image_path
test_paths = df_test.image_path

for filename in df_test.image_id:
    npy_converter(IMAGES_PATH + filename + '.jpg', 512, 512, 'images_npy/' + filename + '.npy')
