# coding=utf-8
import os
import keras
import numpy as np
from keras.preprocessing import image
from model import darknet19


def main():
    # config setting
    image_size = (150, 150, 3)  # width, height, channel
    batch_size = 64
    epochs = 5
    gender_num_classes = 2
    model_dir = os.path.join('..', 'model')
    annotation_file = os.path.join('/home', 'yusuke', 'work', 'data', 'All-Age-Faces_Dataset', 'train_val.txt')
    train_images, val_images = [], []
    train_labels, val_labels = [], []

    # データの格納
    with open(annotation_file, 'r') as inf:
        for line in inf:
            line = line.rstrip()
            vals = line.split(' ')
            if vals[3] == 'train':
                # np.ndarrayでリストに格納
                train_images.append(image.img_to_array(image.load_img(vals[0], target_size=image_size[:2])))
                train_labels.append([keras.utils.to_categorical(np.asarray(int(vals[1])), gender_num_classes),
                                     np.asarray(int(vals[2]))])
            else:
                val_images.append(image.img_to_array(image.load_img(vals[0], target_size=image_size[:2])))
                val_labels.append([keras.utils.to_categorical(np.asarray(int(vals[1])), gender_num_classes),
                                   np.asarray(int(vals[2]))])

    train_images, val_images = np.asarray(train_images), np.asarray(val_images)
    train_images, val_images = train_images / 255, val_images / 255
    print(train_images[0])
    print(train_labels[0])

    model = darknet19(image_size, ['output1', 'output2'], gender_num_classes)
    print(model)


if __name__ == '__main__':
    main()
