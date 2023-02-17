import keras
from lib import CNN_networks

num_classes=10

class FL_Client():

    def __init__(self, model_type, train_images, train_labels, test_images, test_labels):
        self.model_type = model_type
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        (self.train_images, self.train_labels) = self.data_preprocess(self.train_images, self.train_labels)
        (self.test_images, self.test_labels) = self.data_preprocess(self.test_images, self.test_labels)

    def data_preprocess(self, data, labels):
        num_classes = 10
        img_row, img_col, channel = 28, 28, 1
        data = data.reshape(data.shape[0], img_row, img_col, channel)
        data = data.astype("float32")
        data /= 255
        labels = keras.utils.to_categorical(labels, num_classes)
        return (data, labels)

    def train_model(self):
        if(self.model_type=="ResNet"):
            model = CNN_networks.ResNet_model(self.train_images, self.train_labels, self.test_images, self.test_labels)
        elif(self.model_type=="AlexaNet"):
            model = CNN_networks.AlexaNet_model(self.train_images, self.train_labels, self.test_images, self.test_labels)
        elif(self.model_type=="VGG"):
            model = CNN_networks.VGG_model(self.train_images, self.train_labels, self.test_images, self.test_labels)
        elif(self.model_type=="LeNet"):
            model = CNN_networks.LeNet5_model(self.train_images, self.train_labels, self.test_images, self.test_labels)
        elif(self.model_type=="ResNext"):
            model = CNN_networks.ResNext(self.train_images, self.train_labels, self.test_images, self.test_labels)
        else:
            print("Model Type error! Default:ResNext")
            model = CNN_networks.ResNext(self.train_images, self.train_labels, self.test_images, self.test_labels)
        self.model = model


    def reload_model(self, model):
        self.model = model

    def retrain_model(self, train_images, train_labels, batch_size, epochs, verbose):
        self.model.fit(train_images,
                  train_labels,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  validation_data=(self.test_images, self.test_labels),
                  shuffle=True
                  )
        #self.model = model
        return self.model