import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
# Set the color explicitly to black
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 2.5  # You already have this
#this script plots performance when changing C
#for qt_efnn, qt_densenet, qt_resnet, qt_dnn

C_vec=[10,15,20,25]
N=10#plots for N=10
epoch_pattern=r"num_epochs=(\d+)"
std_pattern=r"std_loss=([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)"


def match_line_in_file(test_outFile,epoch_num):
    with open(test_outFile,'r') as fptr:
        contents=fptr.readlines()
    for line in contents:
        # print(line)
        match_epoch=re.search(epoch_pattern,line)
        if match_epoch:
            epoch_in_file=int(match_epoch.group(1))
            if epoch_in_file==epoch_num:
                match_std = re.search(std_pattern,line)
                if match_std:
                    # print(epoch_in_file)
                    # print(float(match_std.group(1)))
                    return epoch_in_file, float(match_std.group(1))
            else:
                continue


def std_loss_all_one_epoch(baseDir,epoch_num,layer,N,C_vec):
    ret_std_loss_vec=[]
    for C in C_vec:
        oneFile=baseDir+f"./out_model_data/N{N}/C{C}/layer{layer}/test_over_epochs.txt"
        # print(f"oneFile={oneFile}")
        ep, stdTmp = match_line_in_file(oneFile, epoch_num)
        # print(f"ep={ep},C={C},layer={layer},sdTmp={stdTmp}")
        ret_std_loss_vec.append(stdTmp)
    return np.array(ret_std_loss_vec)



inDir=f"./train_test_data/N{N}/"
in_pkl_train_file=inDir+"/db.train_num_samples200000.pkl"
with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)
Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

Y_train_avg=np.mean(Y_train_array)

abs_Y_train_avg=np.abs(Y_train_avg)


#load result from qt_efnn
qt_efnn_layer_vec=np.array([0,1,2])
qt_efnn_rate=0.9
qt_efnn_num_suffix=40000
qt_efnn_num_epochs = 500

qt_efnn_baseDir="/home/adada/Documents/pyCode/deep_field/pbc_bn_double_quantum/"

#qt_efnn, 1 layer
qt_efnn_layer0=qt_efnn_layer_vec[0]
qt_efnn_std_loss_layer0_vec=std_loss_all_one_epoch(qt_efnn_baseDir,qt_efnn_num_epochs,qt_efnn_layer0,N,C_vec)
qt_efnn_relative_err_layer0=qt_efnn_std_loss_layer0_vec/abs_Y_train_avg

#qt_efnn, 2 layers
qt_efnn_layer1=qt_efnn_layer_vec[1]
qt_efnn_std_loss_layer1_vec=std_loss_all_one_epoch(qt_efnn_baseDir,qt_efnn_num_epochs,qt_efnn_layer1,N,C_vec)
qt_efnn_relative_err_layer1=qt_efnn_std_loss_layer1_vec/abs_Y_train_avg

#qt_efnn, 3 layers
qt_efnn_layer2=qt_efnn_layer_vec[2]
qt_efnn_std_loss_layer2_vec=std_loss_all_one_epoch(qt_efnn_baseDir,qt_efnn_num_epochs,qt_efnn_layer2,N,C_vec)
qt_efnn_relative_err_layer2=qt_efnn_std_loss_layer2_vec/abs_Y_train_avg

#load result from qt_densenet
qt_densenet_baseDir="/home/adada/Documents/pyCode/deep_field_collection/qt_densenet_pbc/"
qt_densenet_layer_vec=np.array([2,3,4])
qt_densenet_num_suffix=40000
qt_densenet_num_epochs = 500
#qt_densenet, 1 layer
qt_densenet_layer0=qt_densenet_layer_vec[0]
qt_densenet_std_loss_layer0_vec=std_loss_all_one_epoch(qt_densenet_baseDir,qt_densenet_num_epochs,qt_densenet_layer0,N,C_vec)
qt_densenet_relative_err_layer0=qt_densenet_std_loss_layer0_vec/abs_Y_train_avg

#qt_densenet, 2 layers
qt_densenet_layer1=qt_densenet_layer_vec[1]
qt_densenet_std_loss_layer1_vec=std_loss_all_one_epoch(qt_densenet_baseDir,qt_densenet_num_epochs,qt_densenet_layer1,N,C_vec)
qt_densenet_relative_err_layer1=qt_densenet_std_loss_layer1_vec/abs_Y_train_avg

#qt_densenet, 3 layers
qt_densenet_layer2=qt_densenet_layer_vec[2]
qt_densenet_std_loss_layer2_vec=std_loss_all_one_epoch(qt_densenet_baseDir,qt_densenet_num_epochs,qt_densenet_layer2,N,C_vec)
qt_densenet_relative_err_layer2=qt_densenet_std_loss_layer2_vec/abs_Y_train_avg


#load result from qt_resnet
qt_resnet_baseDir="/home/adada/Documents/pyCode/deep_field_collection/qt_resnet_pbc/"
qt_resnet_layer_vec=np.array([1,2,3])
qt_resnet_num_suffix=40000
qt_resnet_num_epochs = 500

#qt_resnet, 1 layer
qt_resnet_layer0=qt_resnet_layer_vec[0]
qt_resnet_std_loss_layer0_vec=std_loss_all_one_epoch(qt_resnet_baseDir,qt_resnet_num_epochs,qt_resnet_layer0,N,C_vec)
qt_resnet_relative_err_layer0=qt_resnet_std_loss_layer0_vec/abs_Y_train_avg

#qt_resnet, 2 layers
qt_resnet_layer1=qt_resnet_layer_vec[1]
qt_resnet_std_loss_layer1_vec=std_loss_all_one_epoch(qt_resnet_baseDir,qt_resnet_num_epochs,qt_resnet_layer1,N,C_vec)
qt_resnet_relative_err_layer1=qt_resnet_std_loss_layer1_vec/abs_Y_train_avg


#qt_resnet, 3 layers
qt_resnet_layer2=qt_resnet_layer_vec[2]
qt_resnet_std_loss_layer2_vec=std_loss_all_one_epoch(qt_resnet_baseDir,qt_resnet_num_epochs,qt_resnet_layer2,N,C_vec)
qt_resnet_relative_err_layer2=qt_resnet_std_loss_layer2_vec/abs_Y_train_avg



#load result from qt_dnn
qt_dnn_baseDir="/home/adada/Documents/pyCode/deep_field_collection/qt_dnn/"
qt_dnn_layer_vec=np.array([1,2,3])
qt_dnn_num_suffix=40000
qt_dnn_num_epochs = 500
#qt_dnn, 1 layer
qt_dnn_layer0=qt_dnn_layer_vec[0]
qt_dnn_std_loss_layer0_vec=std_loss_all_one_epoch(qt_dnn_baseDir,qt_dnn_num_epochs,qt_dnn_layer0,N,C_vec)
qt_dnn_relative_err_layer0=qt_dnn_std_loss_layer0_vec/abs_Y_train_avg


#qt_dnn, 2 layers
qt_dnn_layer1=qt_dnn_layer_vec[1]
qt_dnn_std_loss_layer1_vec=std_loss_all_one_epoch(qt_dnn_baseDir,qt_dnn_num_epochs,qt_dnn_layer1,N,C_vec)
qt_dnn_relative_err_layer1=qt_dnn_std_loss_layer1_vec/abs_Y_train_avg

#qt_dnn, 3 layers
qt_dnn_layer2=qt_dnn_layer_vec[2]
qt_dnn_std_loss_layer2_vec=std_loss_all_one_epoch(qt_dnn_baseDir,qt_dnn_num_epochs,qt_dnn_layer2,N,C_vec)
qt_dnn_relative_err_layer2=qt_dnn_std_loss_layer2_vec/abs_Y_train_avg

