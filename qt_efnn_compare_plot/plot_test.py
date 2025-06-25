import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd


#this script plots test error for
#qt_efnn
#qt_densenet
#qt_resnet
def N_2_test_file(baseDir,N,C,layer_num,num_epochs,num_suffix):
    in_file_dir=baseDir+f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file = in_file_dir + f"/test_epoch{num_epochs}_num_samples{num_suffix}.txt"
    return test_txt_file

def qt_efnn_N_2_test_file(baseDir,N,C,layer_num,num_epochs,num_suffix):
    in_file_dir = baseDir + f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file = in_file_dir + f"/test_over{50}_rate0.9_epoch{num_epochs}_num_samples{num_suffix}.txt"
    return test_txt_file

def N_2_test_data_pkl(baseDir,N,num_suffix):
    in_data_dir=baseDir+f"./larger_lattice_test_performance/N{N}/"
    pkl_test_file = in_data_dir + f"/db.test_num_samples{num_suffix}.pkl"
    return pkl_test_file

pattern_std=r'std_loss=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'


def file_2_std(test_fileName):
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()
    # print(line)
    match_std_loss=re.search(pattern_std,line)
    # match_custom_err=re.search(pattern_custom_err,line)
    if match_std_loss:
        std_loss= float(match_std_loss.group(1))
    else:
        print("format error")
        exit(12)
    # if match_custom_err:
    #     custom_err=float(match_custom_err.group(1))
    return std_loss

C=15
#load result from qt_efnn
qt_efnn_layer_vec=np.array([0,1,2])
qt_efnn_rate=0.9
qt_efnn_N_vec=np.array([10,15,20,25,30,35,40])

qt_efnn_num_suffix=40000
qt_efnn_num_epochs = 500

qt_efnn_baseDir="/home/adada/Documents/pyCode/deep_field/pbc_bn_double_quantum/"

pkl_file_vec=[N_2_test_data_pkl(qt_efnn_baseDir,N,qt_efnn_num_suffix) for N in qt_efnn_N_vec]
abs_avg_Y_train_vec=[]
for j in range(0,len(pkl_file_vec)):
    file_pkl_tmp=pkl_file_vec[j]
    with open(file_pkl_tmp,"rb") as fptr:
        X_train_tmp,Y_train_tmp=pickle.load(fptr)
    Y_train_tmp=np.array(Y_train_tmp)
    absTmp=np.abs(np.mean(Y_train_tmp))
    # absTmp=np.std(Y_train_tmp)
    abs_avg_Y_train_vec.append(absTmp)

abs_avg_Y_train_vec = np.array(abs_avg_Y_train_vec)
#qt_efnn , layer 1
qt_efnn_layer0=qt_efnn_layer_vec[0]
qt_efnn_file_vec_layer0=[qt_efnn_N_2_test_file(qt_efnn_baseDir,N,C,qt_efnn_layer0,qt_efnn_num_epochs,qt_efnn_num_suffix) for N in qt_efnn_N_vec]
qt_efnn_std_loss_vec_layer0=[]
for file in qt_efnn_file_vec_layer0:
    std_loss=file_2_std(file)
    qt_efnn_std_loss_vec_layer0.append(std_loss)
qt_efnn_std_loss_vec_layer0 = np.array(qt_efnn_std_loss_vec_layer0)
qt_efnn_relative_error_layer0=(qt_efnn_std_loss_vec_layer0/abs_avg_Y_train_vec)


#qt_efnn , layer 2
qt_efnn_layer1=qt_efnn_layer_vec[1]
qt_efnn_file_vec_layer1=[qt_efnn_N_2_test_file(qt_efnn_baseDir,N,C,qt_efnn_layer1,qt_efnn_num_epochs,qt_efnn_num_suffix) for N in qt_efnn_N_vec]
qt_efnn_std_loss_vec_layer1=[]
for file in qt_efnn_file_vec_layer1:
    std_loss = file_2_std(file)
    qt_efnn_std_loss_vec_layer1.append(std_loss)

qt_efnn_std_loss_vec_layer1=np.array(qt_efnn_std_loss_vec_layer1)
qt_efnn_relative_error_layer1=(qt_efnn_std_loss_vec_layer1/abs_avg_Y_train_vec)

#qt_efnn , layer 3
qt_efnn_layer2=qt_efnn_layer_vec[2]
qt_efnn_file_vec_layer2=[qt_efnn_N_2_test_file(qt_efnn_baseDir,N,C,qt_efnn_layer2,qt_efnn_num_epochs,qt_efnn_num_suffix) for N in qt_efnn_N_vec]
qt_efnn_std_loss_vec_layer2=[]
for file in qt_efnn_file_vec_layer2:
    std_loss = file_2_std(file)
    qt_efnn_std_loss_vec_layer2.append(std_loss)

qt_efnn_std_loss_vec_layer2=np.array(qt_efnn_std_loss_vec_layer2)
qt_efnn_std_loss_vec_layer2=(qt_efnn_std_loss_vec_layer2/abs_avg_Y_train_vec)


###qt_densenet
qt_densenet_baseDir="/home/adada/Documents/pyCode/deep_field_collection/qt_densenet_pbc/"
qt_densenet_layer_vec=np.array([2,3,4])
qt_densenet_N_vec=np.array([10,15,20,25,30,35,40])
qt_densenet_num_suffix=40000
qt_densenet_num_epochs = 500

#qt_densenet layer1
qt_densenet_layer1=qt_densenet_layer_vec[0]
qt_densenet_file_vec_layer1=[N_2_test_file(qt_densenet_baseDir,N,C,qt_densenet_layer1,qt_densenet_num_epochs,qt_densenet_num_suffix) for N in qt_densenet_N_vec]
qt_densenet_std_loss_vec_layer1=[]
for file in qt_densenet_file_vec_layer1:
    std_loss = file_2_std(file)
    qt_densenet_std_loss_vec_layer1.append(std_loss)

qt_densenet_std_loss_vec_layer1=np.array(qt_densenet_std_loss_vec_layer1)
qt_densenet_relative_error_layer1=(qt_densenet_std_loss_vec_layer1/abs_avg_Y_train_vec)

#qt_densenet layer2
qt_densenet_layer2=qt_densenet_layer_vec[1]
qt_densenet_file_vec_layer2=[N_2_test_file(qt_densenet_baseDir,N,C,qt_densenet_layer2,qt_densenet_num_epochs,qt_densenet_num_suffix) for N in qt_densenet_N_vec]
qt_densenet_std_loss_vec_layer2=[]
for file in qt_densenet_file_vec_layer2:
    std_loss = file_2_std(file)
    qt_densenet_std_loss_vec_layer2.append(std_loss)

qt_densenet_std_loss_vec_layer2=np.array(qt_densenet_std_loss_vec_layer2)
qt_densenet_relative_error_layer2=(qt_densenet_std_loss_vec_layer2/abs_avg_Y_train_vec)

#qt_densenet layer3
qt_densenet_layer3=qt_densenet_layer_vec[2]
qt_densenet_file_vec_layer3=[N_2_test_file(qt_densenet_baseDir,N,C,qt_densenet_layer3,qt_densenet_num_epochs,qt_densenet_num_suffix) for N in qt_densenet_N_vec]
qt_densenet_std_loss_vec_layer3=[]

for file in qt_densenet_file_vec_layer3:
    std_loss = file_2_std(file)
    qt_densenet_std_loss_vec_layer3.append(std_loss)

qt_densenet_std_loss_vec_layer3=np.array(qt_densenet_std_loss_vec_layer3)
qt_densenet_relative_error_layer3=(qt_densenet_std_loss_vec_layer3/abs_avg_Y_train_vec)

#qt_resnet
qt_resnet_baseDir="/home/adada/Documents/pyCode/deep_field_collection/qt_resnet_pbc/"
#qt_resnet layer 1
qt_resnet_layer_vec=np.array([1,2,3])
qt_resnet_N_vec=np.array([10,15,20,25,30,35,40])
qt_resnet_num_suffix=40000
qt_resnet_num_epochs = 500

#qt_resnet layer 1
qt_resnet_layer1=qt_resnet_layer_vec[0]
qt_resnet_file_vec_layer1=[N_2_test_file(qt_resnet_baseDir,N,C,qt_resnet_layer1,qt_resnet_num_epochs,qt_resnet_num_suffix) for N in qt_resnet_N_vec]
qt_resnet_std_loss_vec_layer1=[]
for file in qt_resnet_file_vec_layer1:
    std_loss = file_2_std(file)
    qt_resnet_std_loss_vec_layer1.append(std_loss)

qt_resnet_std_loss_vec_layer1=np.array(qt_resnet_std_loss_vec_layer1)
qt_resnet_relative_error_layer1=(qt_resnet_std_loss_vec_layer1/abs_avg_Y_train_vec)

#qt_resnet layer 2
qt_resnet_layer2=qt_resnet_layer_vec[1]
qt_resnet_file_vec_layer2=[N_2_test_file(qt_resnet_baseDir,N,C,qt_resnet_layer2,qt_resnet_num_epochs,qt_resnet_num_suffix) for N in qt_resnet_N_vec]
qt_resnet_std_loss_vec_layer2=[]

for file in qt_resnet_file_vec_layer2:
    std_loss = file_2_std(file)
    qt_resnet_std_loss_vec_layer2.append(std_loss)

qt_resnet_std_loss_vec_layer2=np.array(qt_resnet_std_loss_vec_layer2)
qt_resnet_relative_error_layer2=(qt_resnet_std_loss_vec_layer2/abs_avg_Y_train_vec)

#qt_resnet layer 3
qt_resnet_layer3=qt_resnet_layer_vec[2]
qt_resnet_file_vec_layer3=[N_2_test_file(qt_resnet_baseDir,N,C,qt_resnet_layer3,qt_resnet_num_epochs,qt_resnet_num_suffix) for N in qt_resnet_N_vec]
qt_resnet_std_loss_vec_layer3=[]
for file in qt_resnet_file_vec_layer3:
    std_loss = file_2_std(file)
    qt_resnet_std_loss_vec_layer3.append(std_loss)

qt_resnet_std_loss_vec_layer3=np.array(qt_resnet_std_loss_vec_layer3)
qt_resnet_relative_error_layer3=(qt_resnet_std_loss_vec_layer3/abs_avg_Y_train_vec)

#plot qt_efnn
plt.figure()

#qt_efnn layer 1
plt.scatter(qt_efnn_N_vec, qt_efnn_relative_error_layer0, color="limegreen", label="EFNN, 1 layer")
plt.plot(qt_efnn_N_vec, qt_efnn_relative_error_layer0, color="limegreen", linestyle="dashed")
#qt_efnn layer 2
plt.scatter(qt_efnn_N_vec, qt_efnn_relative_error_layer1, color="green", label="EFNN, 2 layers")
plt.plot(qt_efnn_N_vec, qt_efnn_relative_error_layer1, color="green", linestyle="dashed")
#qt_efnn layer 3
plt.scatter(qt_efnn_N_vec, qt_efnn_std_loss_vec_layer2, color="darkgreen", label="EFNN, 3 layers")
plt.plot(qt_efnn_N_vec, qt_efnn_std_loss_vec_layer2, color="darkgreen", linestyle="dashed")


#qt_densenet layer 1
plt.scatter(qt_densenet_N_vec, qt_densenet_relative_error_layer1, color="cornflowerblue", label="DenseNet, 1 layer")
plt.plot(qt_densenet_N_vec, qt_densenet_relative_error_layer1, color="cornflowerblue", linestyle="dotted")
#qt_densenet layer 2
plt.scatter(qt_densenet_N_vec, qt_densenet_relative_error_layer2, color="royalblue", label="DenseNet, 2 layers")
plt.plot(qt_densenet_N_vec, qt_densenet_relative_error_layer2, color="royalblue", linestyle="dotted")
#qt_densenet layer 3
plt.scatter(qt_densenet_N_vec, qt_densenet_relative_error_layer3, color="darkblue", label="DenseNet, 3 layers")
plt.plot(qt_densenet_N_vec, qt_densenet_relative_error_layer3, color="darkblue", linestyle="dotted")

#qt_resnet
#qt_resnet layer 1
plt.scatter(qt_resnet_N_vec, qt_resnet_relative_error_layer1, color="indianred", label="ResNet, 1 layer")
plt.plot(qt_resnet_N_vec, qt_resnet_relative_error_layer1, color="indianred", linestyle="dashdot")
#qt_resnet layer 2
plt.scatter(qt_resnet_N_vec, qt_resnet_relative_error_layer2, color="firebrick", label="ResNet, 2 layers")
plt.plot(qt_resnet_N_vec, qt_resnet_relative_error_layer2, color="firebrick", linestyle="dashdot")
#qt_resnet layer 3
plt.scatter(qt_resnet_N_vec, qt_resnet_relative_error_layer3, color="darkred", label="ResNet, 3 layers")
plt.plot(qt_resnet_N_vec, qt_resnet_relative_error_layer3, color="darkred", linestyle="dashdot")
# Add labels and legend
plt.xlabel('Lattice Size (N)', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.title('RKKY', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')
plt.yscale('log')  # Consider log scale if ranges vary widely

plt.tight_layout()
plt.savefig('quantum_nn_comparison.png', dpi=300)
# plt.show()