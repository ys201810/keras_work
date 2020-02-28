# coding=utf-8
import os
import keras
import numpy as np
from keras.preprocessing import image
from model import darknet19
from keras import optimizers
from tensorboard import TrainValTensorBoard
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, BatchNormalization
from keras.models import Model


def main():
    # config setting
    image_size = (150, 150, 3)  # width, height, channel
    batch_size = 64
    epochs = 200
    gender_num_classes = 2
    age_num_classes = 5
    model_dir = os.path.join('..', 'model')
    annotation_file = os.path.join('/home', 'yusuke', 'work', 'data', 'All-Age-Faces_Dataset', 'train_val.txt')
    train_images, val_images = [], []
    train_labels_gender, train_labels_age, val_labels_gender, val_labels_age = [], [], [], []

    training_mode = {'age': 'reg'}  # {'gender': 'cls', 'age': 'cls'}

    # データの格納
    with open(annotation_file, 'r') as inf:
        for line in inf:
            line = line.rstrip()
            vals = line.split(' ')
            if vals[3] == 'train':
                # np.ndarrayでリストに格納
                train_images.append(image.img_to_array(image.load_img(vals[0], target_size=image_size[:2])))
                train_labels_gender.append(keras.utils.to_categorical(np.asarray(int(vals[1])), gender_num_classes))
                if training_mode['age'] == 'cls':
                    if int(vals[2]) <= 18:
                        age = 0
                    elif 18 < int(vals[2]) <= 24:
                        age = 1
                    elif 24 < int(vals[2]) <= 45:
                        age = 2
                    elif 45 < int(vals[2]) <= 65:
                        age = 3
                    elif 65 < int(vals[2]):
                        age = 4
                    train_labels_age.append(keras.utils.to_categorical(np.asarray(age), age_num_classes))
                elif training_mode['age'] == 'reg':
                    train_labels_age.append(np.asarray(int(vals[2])))
            else:
                val_images.append(image.img_to_array(image.load_img(vals[0], target_size=image_size[:2])))
                val_labels_gender.append(keras.utils.to_categorical(np.asarray(int(vals[1])), gender_num_classes))
                if training_mode['age'] == 'cls':
                    if int(vals[2]) <= 18:
                        age = 0
                    elif 18 < int(vals[2]) <= 24:
                        age = 1
                    elif 24 < int(vals[2]) <= 45:
                        age = 2
                    elif 45 < int(vals[2]) <= 65:
                        age = 3
                    elif 65 < int(vals[2]):
                        age = 4
                    val_labels_age.append(keras.utils.to_categorical(np.asarray(age), age_num_classes))
                elif training_mode['age'] == 'reg':
                    val_labels_age.append(np.asarray(int(vals[2])))

    train_images, val_images = np.asarray(train_images), np.asarray(val_images)
    train_images, val_images = train_images / 255, val_images / 255
    train_labels_gender, train_labels_age = np.asarray(train_labels_gender), np.asarray(train_labels_age)
    val_labels_gender, val_labels_age = np.asarray(val_labels_gender), np.asarray(val_labels_age)

    inputs = Input(shape=image_size, name='input')

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_size, name='conv1')(inputs)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    x = Conv2D(64, kernel_size=(1, 1), activation='relu', name='conv4')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv5')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv6')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    x = Conv2D(128, kernel_size=(1, 1), activation='relu', name='conv7')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    x = Conv2D(256, kernel_size=(3, 3),activation='relu', name='conv8')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(512, kernel_size=(3, 3),activation='relu', name='conv9')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    x = Conv2D(256, kernel_size=(1, 1),activation='relu', name='conv10')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    x = Conv2D(512, kernel_size=(3, 3),activation='relu', name='conv11')(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dence1')(x)
    output1 = Dense(gender_num_classes, activation='softmax', name='output1')(x)
    output2 = Dense(1, activation='linear', name='output2')(x)
    output3 = Dense(age_num_classes, activation='softmax', name='output3')(x)

    clbk = TrainValTensorBoard(log_dir=os.path.join(model_dir, 'log'), write_graph=True, histogram_freq=0)

    # シンプルに一つの出力をする時のモデル
    # model = darknet19(image_size, ['output1', 'output2'], gender_num_classes)

    if 'age' in training_mode.keys() and 'gender' in training_mode.keys() and training_mode['age'] == 'reg' and \
            training_mode['gender'] == 'cls':
        # 性別の分類と年齢の回帰を同時学習パターン
        outputs = [output1, output2]
        loss_dict = {'output1': 'categorical_crossentropy', 'output2': 'mean_squared_error'}
        loss_weights = [0.2, 0.8]
        output_data_dict = {'output1': train_labels_gender, 'output2': train_labels_age}
        val_output_data_dict = {'output1': val_labels_gender, 'output2': val_labels_age}
        metrics = ['accuracy']
    elif 'gender' in training_mode.keys() and training_mode['gender'] == 'cls' and 'age' not in training_mode.keys():
        # 性別だけを学習パターン
        outputs = [output1]
        loss_dict = {'output1': 'categorical_crossentropy'}
        loss_weights = [1.0]
        output_data_dict = {'output1': train_labels_gender}
        val_output_data_dict = {'output1': val_labels_gender}
        metrics = ['accuracy']
    elif 'gender' not in training_mode.keys() and 'age' in training_mode.keys() and training_mode['age'] == 'reg':
        # 年齢だけを学習パターン
        outputs = [output2]
        loss_dict = {'output2': 'mse'}
        loss_weights = [1.0]
        output_data_dict = {'output2': train_labels_age}
        val_output_data_dict = {'output2': val_labels_age}
        metrics = ['accuracy', 'mae']
    elif 'gender' not in training_mode.keys() and 'age' in training_mode.keys() and training_mode['age'] == 'cls':
        # 年齢を分類で
        outputs = [output3]
        loss_dict = {'output3': 'categorical_crossentropy'}
        loss_weights = [1.0]
        output_data_dict = {'output3': train_labels_age}
        val_output_data_dict = {'output3': val_labels_age}
        metrics = ['accuracy']
    elif 'gender' in training_mode.keys() and 'age' in training_mode.keys() and training_mode['age'] == 'cls' and \
            training_mode['gender'] == 'cls':
        # 年齢と性別を同時に分類で
        outputs = [output1, output3]
        loss_dict = {'output1': 'categorical_crossentropy', 'output3': 'categorical_crossentropy'}
        loss_weights = [0.4, 0.6]
        output_data_dict = {'output1': train_labels_gender, 'output3': train_labels_age}
        val_output_data_dict = {'output1': val_labels_gender, 'output3': val_labels_age}
        metrics = ['accuracy']

    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(loss=loss_dict, optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  loss_weights=(loss_weights), metrics=metrics)

    history = model.fit({'input': train_images}, output_data_dict, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=({'input': val_images}, val_output_data_dict), callbacks=[clbk])

    score = model.evaluate(val_images, val_output_data_dict, verbose=0)

    model.save_weights(os.path.join(model_dir, 'cnn_model_weights.hdf5'))

    json_string = model.to_json()
    open(os.path.join(model_dir, 'cnn_model_weights.json'), 'w').write(json_string)

    print(history)
    print(score)


if __name__ == '__main__':
    main()
