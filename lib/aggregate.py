from keras.datasets import mnist
import numpy as np
from sklearn.metrics import classification_report
import random
import keras
from keras.datasets import mnist
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pandas as pd
from lib import CNN_networks_2
num_classes = 10
img_row,img_col,channel = 28,28,1


'''-------------------------------------------------------------'''
'''---------------------------功能函数--------------------------'''
def read_local_models(models_addr_list):
    '''
    读取训练好的模型
    models_addr_list是模型存储的地址的列表
    '''
    print('''读取训练好的模型''')
    model_list = []
    for model_addr in models_addr_list:
        model_list.append(load_model(model_addr))
    return model_list

def choose_models(benign_num, untargeted_num,targeted_num, random_num, models_folder_addr):
    '''
    根据指定的benign等数量从指定的models_folder_addr中选择client，并返回对应选择的模型的存储地址列表
    benign_num:选择的benign client的数量
    untargeted_num:选择的untargeted client的数量
    targeted_num:选择的targeted client的数量
    random_num:选择的random client的数量
    models_folder_addr:client模型存储的位置
    '''
    all_model_addr = list_dir(models_folder_addr)
    choosen_model_addr = []
    benign_model_addr = []
    untargeted_model_addr = []
    targeted_model_addr = []
    random_model_addr = []
    for i in all_model_addr:
        if("benign_model" in i):
            benign_model_addr.append(i)
        elif("untargeted_model" in i):
            untargeted_model_addr.append(i)
        elif("targeted_model" in i):
            targeted_model_addr.append(i)
        elif("random_model" in i):
            random_model_addr.append(i)
        else:
            print("read addr error")
    choosen_model_addr = benign_model_addr[:benign_num]+untargeted_model_addr[:untargeted_num]+targeted_model_addr[:targeted_num]+random_model_addr[:random_num]
    return choosen_model_addr

def list_dir(path):
    addr_list = []
    for addr in os.listdir(path):
        if(addr.endswith(".h5")):
            addr_list.append(path+addr)
    return addr_list

def round_get_prediction(models,public_images, model_num):
    '''
    这个函数的作用主要是返回每一轮所有的模型预测结果组成的列表
    :param models: teh clients' models
    :param test_images: the test images
    :param client_num:the number of clients
    :return:the precition of all clients for the epoch [client_num]
    '''
    round_prediction = []#all the clients predictions in this epoch
    for k in range(model_num):
        round_prediction.append(models[k].model.predict(public_images))
    return round_prediction
'''---------------------------功能函数--------------------------'''
'''-------------------------------------------------------------'''



'''-------------------------------------------------------------'''
'''-------------------------数据相关函数------------------------'''
def public_dataset_build(round_num, round_data_num, train_data, train_labels, train_answers):
    '''收集public的dataset'''
    public_set = []#这个是public数据集
    public_labels = []#这个是public标签，经过了预处理，拿给CNN网络用
    real_public_answers = []#这个是public标签，但没有经过预处理，主要用来最后的效果评估
    for i in range(round_num):
        public_set.append(train_data[0+i*1000:800+i*1000])
        public_labels.append(train_labels[0+i*1000:800+i*1000])
        real_public_answers.append(train_answers[0+i*1000:800+i*1000])
    return public_set, public_labels, real_public_answers

def bait_dataset_build(round_num, round_bait_num, test_data, test_labels, test_answers):
    '''制作组成baits的dataset'''
    bait_set = []#这个是鱼饵数据集
    bait_labels = []#这个是鱼饵标签，经过了预处理，拿给CNN网络用
    real_bait_answers = []#这个是鱼饵数据集的真实标签，没有经过预处理，主要用来做最后的byzantine detection
    for i in range(round_num):
        bait_set.append(test_data[800+i*1000:1000+i*1000])
        bait_labels.append(test_labels[800+i*1000:1000+i*1000])
        real_bait_answers.append(test_answers[800+i*1000:1000+i*1000])
    return bait_set, bait_labels, real_bait_answers

def data_preprocess(images, labels):
    '''
    这个函数的作用是将输入的images和labels处理成CNN能够用来训练的格式
    '''
    images = images.reshape(images.shape[0], img_row, img_col, channel)
    images = images.astype("float32")
    images /= 255
    labels = keras.utils.to_categorical(labels, num_classes)
    return images,labels

def read_preprocessed_mnist():
    '''
    读取MNIST数据集并进行预处理
    '''
    print('''读取MNIST数据集并进行预处理''')
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_answers = train_labels
    test_answers = test_labels
    train_data, train_labels = data_preprocess(train_data,train_labels)
    test_data, test_labels = data_preprocess(test_data, test_labels)
    return (train_data, train_labels, train_answers),(test_data, test_labels, test_answers)
'''-------------------------数据相关函数------------------------'''
'''-------------------------------------------------------------'''


'''-------------------------------------------------------------'''
'''-------------------------计算声望函数------------------------'''
def reputation_cal(current_reputation, current_round_num, current_round_accuracy):
    reputation = current_round_num*current_reputation + current_round_accuracy
    reputation /= (current_round_num+1)
    return reputation
'''-------------------------计算声望函数------------------------'''
'''-------------------------------------------------------------'''


'''-------------------------------------------------------------'''
'''---------------------------聚合函数--------------------------'''
def round_aggregate(predictions, round_data_num):
    '''
    对某一round的所有模型返回的预测进行聚合
    '''
    client_num = len(predictions)
    aggregated_prediction = np.zeros((round_data_num, num_classes))
    for k in range(client_num):
        aggregated_prediction += predictions[k]
    aggregated_prediction = np.argmax(aggregated_prediction, axis=1)
    return aggregated_prediction

def round_aggregate_reputation_weighted(predictions, round_data_num, reputations):
    '''
    以reputation为权重，对某一round的所有模型返回的预测进行聚合
    '''
    client_num = len(predictions)
    aggregated_prediction = np.zeros((round_data_num, num_classes))
    for k in range(client_num):
        aggregated_prediction += predictions[k]*reputations[k]
    aggregated_prediction = np.argmax(aggregated_prediction, axis=1)
    return aggregated_prediction

def round_aggregate_reputation_threhold(predictions, round_data_num, reputations):
    '''
    以某个threhold作为界限，当低于这个界限的时候，该model被认为byzantine model，这一轮其对应的pre不要
    '''
    client_num = len(predictions)
    aggregated_prediction = np.zeros((round_data_num, num_classes))
    for k in range(client_num):
        if(reputations[k]>threhold):
            aggregated_prediction += predictions[k]
    aggregated_prediction = np.argmax(aggregated_prediction, axis=1)
    return aggregated_prediction
'''---------------------------聚合函数--------------------------'''
'''-------------------------------------------------------------'''



'''-----------------------------------------------------------------------'''
'''-------------------------模拟服务器训练模型函数------------------------'''
def get_all_acc(server_set, all_answers):
    '''
    模拟server模型获得aggregated-server_set后的训练过程
    server_set:获得的aggregated的所有轮次的public_dataset
    all_answers:aggregated得到的server_set对应的标签
    '''
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    test_answers = test_labels
    test_data, test_labels = data_preprocess(test_data, test_labels)
    all_acc = []

    model = CNN_networks_2.Res_Build()
    for i in range(len(server_set)):
        model = CNN_networks_2.ResNet_train(model, server_set[i], all_answers[i], test_data[8000:10000], test_labels[8000:10000])
        test_prediction = model.predict(test_data[8000:10000])
        test_prediction = np.argmax(test_prediction, axis=1)
        all_acc.append(accuracy_score(test_answers[8000:10000], test_prediction))
    return all_acc


def retrain_model(fl_server, train_images, train_labels, test_images, test_labels):
    history = fl_server.model.fit(train_images, train_labels, batch_size=50, epochs=10, verbose=1, 
                        validation_data=(test_images, test_labels))
    score = fl_server.model.evaluate(test_images, test_labels, verbose=0)
    return self.model
'''-------------------------模拟服务器训练模型函数------------------------'''
'''-----------------------------------------------------------------------'''


def all_aggregate(round_num, models_addr_list, FL_weight, FL_threhold, round_public_size=800, round_bait_size=200):
    #获取处理过的数据集
    (train_data, train_labels, train_answers),(test_data, test_labels, test_answers)=read_preprocessed_mnist()
    #获取存储的已训练好的local模型
    model_list = read_local_models(models_addr_list)

    '''读取数据，public的数据是server用来发给client的模型打标签的数据;bait的数据是server用来发给client的模型进行钓鱼的数据'''
    public_set, public_labels, real_public_answers = public_dataset_build(round_num, round_public_size, test_data, test_labels, test_answers)
    bait_set, bait_labels, real_bait_answers = bait_dataset_build(round_num, round_bait_size, test_data, test_labels, test_answers)

    
    '''used数据是对public数据和bait数据进行组合后的数据'''
    used_set = []
    used_labels = []
    real_used_answers = []
    for m in range(round_num):
        used_set.append(np.vstack((bait_set[m],public_set[m])))
        used_labels.append(np.vstack((bait_labels[m],public_labels[m])))
        real_used_answers.append(np.hstack((real_bait_answers[m],real_public_answers[m])))
    
    
    print('''初始化声望机制的各项指标''')
    round_reputations = [0.5]*len(models_addr_list)#round_reputations指的是当前的round的reputations
    all_reputations = []#all_reputations指的是所有round的round_reputations的集合
    all_reputations.append(round_reputations)

    
    print('''开始进行集成联邦学习''')
    all_used_answers = []
    all_aggregated_answers = []
    for i in range(round_num):
        round_used_answers = []
        round_reputations = []
        #进行每一轮的学习
        for k in range(len(model_list)):
            round_used_answers.append(model_list[k].predict(used_set[i]))
            #计算每一个client对bait_set的预测结果的准确度
            current_acc = accuracy_score(real_bait_answers[i], np.argmax(round_used_answers[k][:round_bait_size], axis=1))
            #当前轮第k个client的reputation并存放到当前轮的round_reputations中
            round_reputations.append(reputation_cal(all_reputations[i][k],i+1, current_acc))
            #去掉当前轮次预测结果中的bait_set部分的预测（没用）
            round_used_answers[k] = round_used_answers[k][round_bait_size:round_bait_size+round_public_size]
        all_reputations.append(round_reputations)
        all_used_answers.append(round_used_answers)
        
        '''根据输入的参数选择聚合算法'''
        if(FL_weight==1):
            round_aggregated_answers = round_aggregate_reputation_weighted(round_used_answers,round_public_size, round_reputations)
        elif(FL_threhold==1):
            round_aggregated_answers = round_aggregate_reputation_threhold(round_used_answers,round_public_size, round_reputations)
        else:
            round_aggregated_answers = round_aggregate(round_used_answers,round_public_size)
        all_aggregated_answers.append(round_aggregated_answers)
    return all_reputations, all_aggregated_answers, public_set, public_labels, real_public_answers