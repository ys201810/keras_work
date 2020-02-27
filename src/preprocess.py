# coding=utf-8
import os
import cv2


def resize_save_images(target_files, image_path, out_path, resize_size):
    for target_file in target_files:
        with open(target_file, 'r') as inf:
            for line in inf:
                line = line.rstrip()
                vals = line.split(' ')
                img = cv2.imread(os.path.join(image_path, vals[0]))
                resize_img = cv2.resize(img, (int(resize_size[0] / 3), int(resize_size[1] / 3)))
                resize_img = cv2.resize(resize_img, resize_size)
                cv2.imwrite(os.path.join(out_path, vals[0].replace('.jpg', '_resize.jpg')), resize_img)


def make_train_annotation(target_files, annotation_file, resized_path):
    if os.path.exists(annotation_file):
        os.remove(annotation_file)

    with open(annotation_file, 'a') as outf:
        for target_file in target_files:
            with open(target_file, 'r') as inf:
                for line in inf:
                    line = line.rstrip()
                    vals = line.split(' ')
                    if target_file.find('train') != -1:
                        # print(' '.join([resized_path + vals[0].replace('.jpg', '_resize.jpg'),
                        #                      vals[1], vals[0][6:8]]))
                        outf.write(' '.join([os.path.join(resized_path, vals[0].replace('.jpg', '_resize.jpg')),
                                             vals[1], str(int(vals[0][6:8])), 'train']) + '\n')
                    else:
                        outf.write(' '.join([os.path.join(resized_path, vals[0].replace('.jpg', '_resize.jpg')),
                                             vals[1], str(int(vals[0][6:8])), 'val']) + '\n')



def main():
    # 画像を同じサイズに整える
    data_root = os.path.join('/home', 'yusuke', 'work', 'data', 'All-Age-Faces_Dataset')
    train_annotation_file = os.path.join(data_root, 'image_sets', 'train.txt')
    val_annotation_file = os.path.join(data_root, 'image_sets', 'val.txt')
    image_path = os.path.join(data_root, 'original_images')
    resized_path = os.path.join(data_root, 'resize_images')

    resize_size = (150, 150)
    # resize_save_images([train_annotation_file, val_annotation_file], image_path, resized_path, resize_size)

    # 学習用アノテーションファイルを用意する。
    annotation_file = os.path.join(data_root, 'train_val.txt')
    make_train_annotation([train_annotation_file, val_annotation_file], annotation_file, resized_path)

if __name__ == '__main__':
    main()
