from keras.datasets import mnist
import numpy as np
from sklearn.metrics import classification_report
#from lib import aggregate
import random
import keras
from keras.datasets import mnist
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pandas as pd
num_classes = 10
img_row,img_col,channel = 28,28,1
random.seed(1)

threhold = 0.5

def data_preprocess(images, labels):
    '''
    这个函数的作用是将输入的images和labels处理成CNN能够用来训练的格式
    '''
    images = images.reshape(images.shape[0], img_row, img_col, channel)
    images = images.astype("float32")
    images /= 255
    labels = keras.utils.to_categorical(labels, num_classes)
    return images,labels

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

def list_dir(path):
    addr_list = []
    for addr in os.listdir(path):
        if(addr.endswith(".h5")):
            addr_list.append(path+addr)
    return addr_list

def round_aggregate_reputation_weighted_pro(predictions, round_data_num, reputations):
    '''
    对每一个client的每一个类型的数据分配一个声望reputation，这个声望由这个client对该类型数据的accuracy和callbacks相乘得来
    '''
    client_num = len(predictions)
    aggregated_prediction = np.zeros((round_data_num, num_classes))
    for k in range(client_num):
        aggregated_prediction += predictions[k]*np.array(reputations[k])
    aggregated_prediction = np.argmax(aggregated_prediction, axis=1)
    return aggregated_prediction

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

def bait_dataset_divide(round_num, round_bait_num, test_data, test_labels, test_answers):
    '''制作组成baits的dataset'''
    bait_set = []#这个是鱼饵数据集
    bait_labels = []#这个是鱼饵标签，经过了预处理，拿给CNN网络用
    real_bait_answers = []#这个是鱼饵数据集的真实标签，没有经过预处理，主要用来做最后的byzantine detection
    
    
    index_list = []
    for i in range(round_num):
        bait_set.append(test_data[800+i*1000:1000+i*1000])
        bait_labels.append(test_labels[800+i*1000:1000+i*1000])
        real_bait_answers.append(test_answers[800+i*1000:1000+i*1000])
        temp_index_list = []
        for n in range(10):
            temp_index_list.append(list(np.where(real_bait_answers[i]==n)))
        #print(temp_index_list)
        index_list.append(temp_index_list)
    return bait_set, bait_labels, real_bait_answers,index_list


def get_all_acc(server_set, all_answers):
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

def reputation_draw(all_reputations, untargeted_num, random_num, targeted_num, benign_num, csv_addr):
    all_reputations = np.array(all_reputations)
    all_reputations = pd.DataFrame(all_reputations)
    all_reputations.to_csv(csv_addr, index=None)
    begin_index = 0
    untargeted_reputations = all_reputations.iloc[:,0:untargeted_num]
    begin_index+=untargeted_num
    random_reputations = all_reputations.iloc[:,begin_index:begin_index+random_num]
    begin_index+=random_num
    targeted_reputations = all_reputations.iloc[:,begin_index:begin_index+targeted_num]
    begin_index+=targeted_num
    benign_reputations = all_reputations.iloc[:,begin_index:begin_index+benign_num]
    benign_reputations['mean_reputation'] = benign_reputations.mean(axis=1)
    untargeted_reputations['mean_reputation'] = untargeted_reputations.mean(axis=1)
    random_reputations['mean_reputation'] = random_reputations.mean(axis=1)
    targeted_reputations['mean_reputation'] = targeted_reputations.mean(axis=1)
    benign_reputations['mean_reputation'].plot()
    untargeted_reputations['mean_reputation'].plot()
    targeted_reputations['mean_reputation'].plot()
    random_reputations['mean_reputation'].plot()
    

def choose_models(benign_num, untargeted_num,targeted_num, random_num, models_folder_addr):
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

def reputation_cal(current_reputation, current_round_num, current_round_accuracy):
    reputation = current_round_num*current_reputation + current_round_accuracy
    reputation /= (current_round_num+1)
    return reputation

def reputation_cal_pro(former_reputations,current_round_num,current_bait_index,model,bait_set,real_answer):
    current_reputations = []
    current_answer = np.argmax(model.predict(bait_set))
    for i in range(10):
        current_acc = accuracy_score(real_answer[current_bait_index[i]], real_answer[current_bait_index[i]])
        current_rep = reputation_cal(former_reputations[i],current_round_num,current_acc)
        current_reputations.append(current_rep)
    return current_reputations