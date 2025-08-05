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

#######
width=6
height=8
textSize=25
yTickSize=33
xTickSize=33
legend_fontsize=23
marker_size1=100
marker_size2=80
lineWidth1=3
lineWidth2=2
tick_length=13
tick_width=2
minor_tick_length=7
minor_tick_width=1


plt.figure(figsize=(width, height))
plt.minorticks_on()
#qt_efnn layer 1
line1=plt.scatter(C_vec,qt_efnn_relative_err_layer0,color="green", label="EFNN",s=marker_size2)
plt.plot(C_vec,qt_efnn_relative_err_layer0,color="green", linestyle="dashed",linewidth=lineWidth2)


#qt_densenet layer 1
line4=plt.scatter(C_vec,qt_densenet_relative_err_layer0,marker="s", color="red", label="DenseNet",s=marker_size2)
plt.plot(C_vec,qt_densenet_relative_err_layer0, color="red", linestyle="dashed", linewidth=lineWidth2)


#qt_resnet layer 1
line7=plt.scatter(C_vec,qt_resnet_relative_err_layer0,marker="^", color="purple", label="ResNet", s=marker_size2)
plt.plot(C_vec,qt_resnet_relative_err_layer0,color="purple", linestyle="dashed", linewidth=lineWidth2)

#qt_dnn layer1
line10 = plt.scatter(C_vec,qt_dnn_relative_err_layer0,marker="D", color="blue", label="DNN", s=marker_size2)

plt.plot(C_vec,qt_dnn_relative_err_layer0,color="blue", linestyle="dashed",linewidth=lineWidth2)


lin_mean_mse=0.9847254886017068
lin_mean_std=np.sqrt(lin_mean_mse)
lin_err_relative=lin_mean_std/abs_Y_train_avg
plt.axhline(y=lin_err_relative, color="black", linestyle="--", label=f"Effective model",linewidth=lineWidth1)


plt.xlabel('Channel Number (C)',fontsize=textSize)
plt.ylabel('Relative Error',fontsize=textSize)

plt.tick_params(axis='both', length=tick_length, width=tick_width)
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width)

plt.xticks([10,15,20,25],labels=["10","15","20","25"],fontsize=xTickSize)
plt.yticks(fontsize=yTickSize)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
legend = plt.legend(loc="best", fontsize=legend_fontsize-10,
          ncol=2,
          handlelength=0.8,
          handletextpad=0.2,
          columnspacing=0.3,
          borderpad=0.1,
          labelspacing=0.15,
          markerscale=0.6)

plt.yscale('log')  # Consider log scale if ranges vary widely
plt.xticks()
plt.yticks(fontsize=yTickSize)
plt.subplots_adjust(left=0.3, right=0.95, top=0.99, bottom=0.15)
plt.savefig('1layer_qt_efnn_vs_all_C', dpi=300)
plt.savefig('1layer_qt_efnn_vs_all_C.svg')
plt.close()