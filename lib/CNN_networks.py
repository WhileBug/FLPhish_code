from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
from keras.layers import *


num_classes=10



def ResNet_model(train_images, train_labels, test_images, test_labels):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(81, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    #model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, batch_size=32, epochs=10, verbose=1,
                        validation_data=(test_images, test_labels))
    score = model.evaluate(test_images, test_labels, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model


def AlexaNet_model(train_images, train_labels, test_images, test_labels):
    model = Sequential()
    '''
    #第一个卷积层
    input_shape  输入平面
    filters 卷积核个数
    kernal_size 卷积窗口大小
    stride 步长
    padding  same/valid
    activation 激活函数
    '''
    model.add(Convolution2D(
        input_shape=(28, 28, 1),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation='relu',

    ))
    # 第一个池化层
    model.add(MaxPool2D(
        pool_size=2,
        strides=2,
        padding='same',
    ))
    # 第二个卷积层
    model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))
    # 第二个卷积层
    model.add(MaxPool2D(2, 2, 'same'))
    # 把第二个卷积层输出扁平化为1维
    model.add(Flatten())
    # 第一个全连接层
    model.add(Dense(1024, activation='relu'))
    # dropout
    model.add(Dropout(0.5))
    # 第二个全连接层
    model.add(Dense(10, activation='softmax'))
    # 定义优化器
    adam = Adam(lr=1e-4)
    # 定义优化器，loss  function
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(train_images, train_labels, batch_size=64, epochs=10)
    # 评估模型
    loss, accuracy = model.evaluate(test_images, test_labels)

    print('test loss', loss)
    print("test accuray", accuracy)
    return model


def VGG_model(train_images, train_labels, test_images, test_labels):
    ishape = 224
    model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(ishape, ishape, 3))

    for layer in model_vgg.layers:
        layer.trainable = False
    model = Flatten()(model_vgg.output)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(0.5)(model)
    model = Dense(10, activation='softmax', name='prediction')(model)
    model_vgg_mnist_pretrain = Model(model_vgg.input, model, name='vgg16_pretrain')
    #print(model_vgg_mnist_pretrain.summary())

    ##我们只需要训练25万个参数，比之前整数少了60倍。
    sgd = SGD(lr=0.05, decay=1e-5)
    model_vgg_mnist_pretrain.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model_vgg_mnist_pretrain.fit(train_images, train_labels,
                                 validation_data=(test_images, test_labels), epochs=10, batch_size=64)

    #######在测试集上评价模型精确度
    scores = model_vgg_mnist_pretrain.evaluate(test_images, test_labels, verbose=0)

    #####打印精确度
    print(scores)
    return model


def LeNet5_model(train_images, train_labels, test_images, test_labels):
    model = Sequential()
    model.add(
        Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), activation='tanh'))  # C1
    model.add(MaxPooling2D(pool_size=(2, 2)))  # S2
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'))  # C3
    model.add(MaxPooling2D(pool_size=(2, 2)))  # S4
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))  # C5
    model.add(Dense(84, activation='tanh'))  # F6
    model.add(Dense(10, activation='softmax'))  # output
    #model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, batch_size=500, epochs=50, verbose=1,
                        validation_data=(test_images, test_labels))
    score = model.evaluate(test_images, test_labels, verbose=0)
    print(score)
    return model


def densenet(x):
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x2 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(1, 1))(x1)

    x3 = concatenate([x1, x2], axis=3)
    x = BatchNormalization()(x3)
    x = Activation('relu')(x)
    x4 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)

    x5 = concatenate([x3, x4], axis=3)
    x = BatchNormalization()(x5)
    x = Activation('relu')(x)
    x6 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)

    x7 = concatenate([x5, x6], axis=3)
    x = BatchNormalization()(x7)
    x = Activation('relu')(x)
    x8 = Conv2D(124, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)

    x = BatchNormalization()(x8)
    x = Activation('relu')(x)
    x9 = Conv2D(124, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
    x9 = MaxPooling2D(pool_size=(2, 2))(x9)
    return x9


def ResNext(train_images, train_labels, test_images, test_labels):
    inputs = Input(shape=(28, 28, 1))

    x = densenet(inputs)
    x = densenet(x)
    x = densenet(x)

    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='sigmoid')(x)

    # 确定模型
    model = Model(inputs=inputs, outputs=x)
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, batch_size=64,
              validation_data=(test_images, test_labels), shuffle=True)
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model

