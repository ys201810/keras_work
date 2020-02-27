# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, BatchNormalization


def darknet19(input_shape, outputs, num_classes):
    inputs = Input(shape=input_shape)
    model = Sequential()

    x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name='conv1')(inputs)
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

    # x = Conv2D(256, kernel_size=(1, 1),activation='relu', name='conv12')(x)
    # x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    # x = Conv2D(512, kernel_size=(3, 3),activation='relu', name='conv13')(x)
    # x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # x = Conv2D(1024, kernel_size=(3, 3),activation='relu', name='conv14')(x)
    # x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    # x = Conv2D(512, kernel_size=(1, 1),activation='relu', name='conv15')(x)
    # x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    # x = Conv2D(1024, kernel_size=(1, 1),activation='relu', name='conv17')(x)
    # x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dence1')(x)
    output1 = Dense(num_classes, activation='softmax', name='output1')(x)
    output2 = Dense(1, activation='sigmoid', name='output2')(x)
    # for layer in model.layers:
        # print(layer.get_output_at(0).get_shape().as_list())
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=input_shape, name='conv1'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', name='conv2'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, kernel_size=(3, 3),activation='relu', name='conv3'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(64, kernel_size=(1, 1),activation='relu', name='conv4'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(128, kernel_size=(3, 3),activation='relu', name='conv5'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(256, kernel_size=(3, 3),activation='relu', name='conv6'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(128, kernel_size=(1, 1),activation='relu', name='conv7'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(256, kernel_size=(3, 3),activation='relu', name='conv8'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(512, kernel_size=(3, 3),activation='relu', name='conv9'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(256, kernel_size=(1, 1),activation='relu', name='conv10'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(512, kernel_size=(3, 3),activation='relu', name='conv11'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(256, kernel_size=(1, 1),activation='relu', name='conv12'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(512, kernel_size=(3, 3),activation='relu', name='conv13'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(1024, kernel_size=(3, 3),activation='relu', name='conv14'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Conv2D(512, kernel_size=(1, 1),activation='relu', name='conv15'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    # memory lack....
    # model.add(Conv2D(1024, kernel_size=(3, 3),activation='relu', name='conv16'))
    # model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    # model.add(Conv2D(512, kernel_size=(1, 1),activation='relu', name='conv17')) # original = 512
    model.add(Conv2D(1024, kernel_size=(1, 1),activation='relu', name='conv17'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    # memory lack....
    # model.add(Conv2D(1024, kernel_size=(3, 3),activation='relu', name='conv18'))
    # model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name='dence1'))
    model.add(Dense(num_classes, activation='softmax', name='output1'))
    model.add(Dense(1, activation='sigmoid', name='output2'))

    # for layer in model.layers:
        # print(layer.get_output_at(0).get_shape().as_list())
    """
    return model
