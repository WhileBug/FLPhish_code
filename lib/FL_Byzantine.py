from lib.FL_Client import FL_Client
import random
import numpy as np
import keras
num_classes=10

class FL_Byzantine(FL_Client):
    def __init__(self, model_type, train_images, train_labels, test_images, test_labels, attack_mode="untargeted"):
        FL_Client.__init__(self, model_type, train_images, train_labels, test_images, test_labels)
        self.train_labels = np.argmax(self.train_labels, axis=1)
        self.test_labels = np.argmax(self.test_labels, axis=1)
        self.attack_mode = attack_mode
        if(self.attack_mode=="untargeted"):
            self.untargeted_attack()
        elif(self.attack_mode=='random'):
            self.random_attack()
        elif(self.attack_mode=="targeted"):
            self.targeted_attack()
        else:
            print("attack_mode error, remain benign")
        self.train_labels = keras.utils.to_categorical(self.train_labels, num_classes)
        self.test_labels = keras.utils.to_categorical(self.test_labels, num_classes)
    def untargeted_attack(self):
        '''
        This type of attack will mislabel the local data from i to (i-5)%10
        '''
        for i in range(len(self.train_labels)):
            self.train_labels[i] = (self.train_labels[i]-5)%10
        for i in range(len(self.test_labels)):
            self.test_labels[i] = (self.test_labels[i]-5)%10
    def random_attack(self):
        '''
        This type of attack will mislabel the local data randomly
        '''
        for i in range(len(self.train_labels)):
            self.train_labels[i] = random.randint(0,9)
        for i in range(len(self.test_labels)):
            self.test_labels[i] = random.randint(0,9)
    def targeted_attack(self):
        '''
        thsi type of attack will mislabel the local data 7 to 1
        '''
        for i in range(len(self.train_labels)):
            if(self.train_labels[i]==7):
                self.train_labels[i] = 1
        for i in range(len(self.test_labels)):
            if(self.test_labels[i]==7):
                self.test_labels[i] = 1