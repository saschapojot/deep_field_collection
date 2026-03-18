import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from itertools import combinations
import matplotlib as mpl
#this script plots test error for
#qt_efnn, qt_dnn, qt_densenet , qt_resnet
# Set the color explicitly to black
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 2.5  # You already have this
def N_2_test_file(baseDir,N,C,layer_num,num_epochs,num_suffix):
    in_file_dir=baseDir+f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file = in_file_dir + f"/test_epoch{num_epochs}_num_samples{num_suffix}.txt"
    return test_txt_file

def qt_efnn_N_2_test_file(baseDir,N,C,layer_num,num_epochs,num_suffix):
    in_file_dir = baseDir + f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file = in_file_dir + f"/test_epoch{num_epochs}_num_samples40000.txt"
    return test_txt_file

def N_2_test_data_pkl(baseDir,N,num_suffix):
    in_data_dir=baseDir+f"./larger_lattice_test_performance/N{N}/"
    pkl_test_file = in_data_dir + f"/db.test_num_samples{num_suffix}.pkl"
    return pkl_test_file


pattern_std=r'std_loss=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
pattern_num_params=r"num_params\s*=\s*(\d+)"

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
def file_2_num_params(test_fileName):
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()
    match_num_params = re.search(pattern_num_params, line)
    if match_num_params:
        return int(match_num_params.group(1))
    else:
        print(f"{test_fileName}, format error")
        exit(12)
C=3
set_epoch=1000
#load result from qt_efnn
qt_efnn_layer_vec=np.array([1,2,3])
qt_efnn_rate=0.9
qt_efnn_N_vec=np.array([10,15,20,25,30,35,40])

qt_efnn_num_suffix=40000
qt_efnn_num_epochs = set_epoch

qt_efnn_baseDir="/home/adada/Documents/pyCode/deep_field_collection/final_sum_qt_efnn/"

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
qt_efnn_num_params_vec_layer0=[]
for file in qt_efnn_file_vec_layer0:
    std_loss=file_2_std(file)
    num_params=file_2_num_params(file)
    qt_efnn_std_loss_vec_layer0.append(std_loss)
    qt_efnn_num_params_vec_layer0.append(num_params)
qt_efnn_std_loss_vec_layer0 = np.array(qt_efnn_std_loss_vec_layer0)
qt_efnn_relative_error_layer0=(qt_efnn_std_loss_vec_layer0/abs_avg_Y_train_vec)
# print(f"qt_efnn_num_params_vec_layer0={qt_efnn_num_params_vec_layer0}")
# print(f"qt_efnn_relative_error_layer0={qt_efnn_relative_error_layer0}")
#qt_efnn , layer 2
qt_efnn_layer1=qt_efnn_layer_vec[1]
qt_efnn_file_vec_layer1=[qt_efnn_N_2_test_file(qt_efnn_baseDir,N,C,qt_efnn_layer1,qt_efnn_num_epochs,qt_efnn_num_suffix) for N in qt_efnn_N_vec]
qt_efnn_std_loss_vec_layer1=[]
qt_efnn_num_params_vec_layer1=[]
for file in qt_efnn_file_vec_layer1:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_efnn_std_loss_vec_layer1.append(std_loss)
    qt_efnn_num_params_vec_layer1.append(num_params)

qt_efnn_std_loss_vec_layer1=np.array(qt_efnn_std_loss_vec_layer1)
qt_efnn_relative_error_layer1=(qt_efnn_std_loss_vec_layer1/abs_avg_Y_train_vec)
# print(f"qt_efnn_num_params_vec_layer1={qt_efnn_num_params_vec_layer1}")
# print(f"qt_efnn_relative_error_layer1={qt_efnn_relative_error_layer1}")
#qt_efnn , layer 3
qt_efnn_layer2=qt_efnn_layer_vec[2]
qt_efnn_file_vec_layer2=[qt_efnn_N_2_test_file(qt_efnn_baseDir,N,C,qt_efnn_layer2,qt_efnn_num_epochs,qt_efnn_num_suffix) for N in qt_efnn_N_vec]
qt_efnn_std_loss_vec_layer2=[]
qt_efnn_num_params_vec_layer2=[]
for file in qt_efnn_file_vec_layer2:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_efnn_std_loss_vec_layer2.append(std_loss)
    qt_efnn_num_params_vec_layer2.append(num_params)

qt_efnn_std_loss_vec_layer2=np.array(qt_efnn_std_loss_vec_layer2)
qt_efnn_relative_error_vec_layer2=(qt_efnn_std_loss_vec_layer2/abs_avg_Y_train_vec)
print(f"qt_efnn_relative_error_vec_layer2={qt_efnn_relative_error_vec_layer2}")


###qt_densenet
qt_densenet_baseDir="/home/adada/Documents/pyCode/deep_field_collection/final_sum_qt_densenet/"
qt_densenet_layer_vec=np.array([1,2,3])
qt_densenet_N_vec=np.array([10,15,20,25,30,35,40])
qt_densenet_num_suffix=40000
qt_densenet_num_epochs = set_epoch

#qt_densenet layer1
qt_densenet_layer1=qt_densenet_layer_vec[0]
qt_densenet_file_vec_layer1=[N_2_test_file(qt_densenet_baseDir,N,C,qt_densenet_layer1,qt_densenet_num_epochs,qt_densenet_num_suffix) for N in qt_densenet_N_vec]
qt_densenet_std_loss_vec_layer1=[]
qt_densenet_num_params_vec_layer1=[]
for file in qt_densenet_file_vec_layer1:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_densenet_std_loss_vec_layer1.append(std_loss)
    qt_densenet_num_params_vec_layer1.append(num_params)

qt_densenet_std_loss_vec_layer1=np.array(qt_densenet_std_loss_vec_layer1)
qt_densenet_relative_error_layer1=(qt_densenet_std_loss_vec_layer1/abs_avg_Y_train_vec)

#qt_densenet layer2
qt_densenet_layer2=qt_densenet_layer_vec[1]
qt_densenet_file_vec_layer2=[N_2_test_file(qt_densenet_baseDir,N,C,qt_densenet_layer2,qt_densenet_num_epochs,qt_densenet_num_suffix) for N in qt_densenet_N_vec]
qt_densenet_std_loss_vec_layer2=[]
qt_densenet_num_params_vec_layer2=[]
for file in qt_densenet_file_vec_layer2:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_densenet_std_loss_vec_layer2.append(std_loss)
    qt_densenet_num_params_vec_layer2.append(num_params)

qt_densenet_std_loss_vec_layer2=np.array(qt_densenet_std_loss_vec_layer2)
qt_densenet_relative_error_layer2=(qt_densenet_std_loss_vec_layer2/abs_avg_Y_train_vec)

#qt_densenet layer3
qt_densenet_layer3=qt_densenet_layer_vec[2]
qt_densenet_file_vec_layer3=[N_2_test_file(qt_densenet_baseDir,N,C,qt_densenet_layer3,qt_densenet_num_epochs,qt_densenet_num_suffix) for N in qt_densenet_N_vec]
qt_densenet_std_loss_vec_layer3=[]
qt_densenet_num_params_vec_layer3=[]
for file in qt_densenet_file_vec_layer3:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_densenet_std_loss_vec_layer3.append(std_loss)
    qt_densenet_num_params_vec_layer3.append(num_params)

qt_densenet_std_loss_vec_layer3=np.array(qt_densenet_std_loss_vec_layer3)
qt_densenet_relative_error_layer3=(qt_densenet_std_loss_vec_layer3/abs_avg_Y_train_vec)

#qt_resnet
qt_resnet_baseDir="/home/adada/Documents/pyCode/deep_field_collection/final_sum_qt_resnet/"

qt_resnet_layer_vec=np.array([1,2,3])
qt_resnet_N_vec=np.array([10,15,20,25,30,35,40])
qt_resnet_num_suffix=40000
qt_resnet_num_epochs = set_epoch

#qt_resnet layer 1
qt_resnet_layer1=qt_resnet_layer_vec[0]
qt_resnet_file_vec_layer1=[N_2_test_file(qt_resnet_baseDir,N,C,qt_resnet_layer1,qt_resnet_num_epochs,qt_resnet_num_suffix) for N in qt_resnet_N_vec]
qt_resnet_std_loss_vec_layer1=[]
qt_resnet_num_params_vec_layer1=[]
for file in qt_resnet_file_vec_layer1:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_resnet_std_loss_vec_layer1.append(std_loss)
    qt_resnet_num_params_vec_layer1.append(num_params)

qt_resnet_std_loss_vec_layer1=np.array(qt_resnet_std_loss_vec_layer1)
qt_resnet_relative_error_layer1=(qt_resnet_std_loss_vec_layer1/abs_avg_Y_train_vec)

#qt_resnet layer 2
qt_resnet_layer2=qt_resnet_layer_vec[1]
qt_resnet_file_vec_layer2=[N_2_test_file(qt_resnet_baseDir,N,C,qt_resnet_layer2,qt_resnet_num_epochs,qt_resnet_num_suffix) for N in qt_resnet_N_vec]
qt_resnet_std_loss_vec_layer2=[]
qt_resnet_num_params_vec_layer2=[]
for file in qt_resnet_file_vec_layer2:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_resnet_std_loss_vec_layer2.append(std_loss)
    qt_resnet_num_params_vec_layer2.append(num_params)

qt_resnet_std_loss_vec_layer2=np.array(qt_resnet_std_loss_vec_layer2)
qt_resnet_relative_error_layer2=(qt_resnet_std_loss_vec_layer2/abs_avg_Y_train_vec)

#qt_resnet layer 3
qt_resnet_layer3=qt_resnet_layer_vec[2]
qt_resnet_file_vec_layer3=[N_2_test_file(qt_resnet_baseDir,N,C,qt_resnet_layer3,qt_resnet_num_epochs,qt_resnet_num_suffix) for N in qt_resnet_N_vec]
qt_resnet_std_loss_vec_layer3=[]
qt_resnet_num_params_vec_layer3=[]
for file in qt_resnet_file_vec_layer3:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_resnet_std_loss_vec_layer3.append(std_loss)
    qt_resnet_num_params_vec_layer3.append(num_params)

qt_resnet_std_loss_vec_layer3=np.array(qt_resnet_std_loss_vec_layer3)
qt_resnet_relative_error_layer3=(qt_resnet_std_loss_vec_layer3/abs_avg_Y_train_vec)

#qt_dnn
qt_dnn_baseDir="/home/adada/Documents/pyCode/deep_field_collection/final_sum_qt_dnn/"
qt_dnn_layer_vec=np.array([1,2,3])
qt_dnn_N_vec=np.array([10,15,20,25,30,35,40])
qt_dnn_num_suffix=40000
qt_dnn_num_epochs = set_epoch

#qt_dnn layer 1
qt_dnn_layer1=qt_dnn_layer_vec[0]
qt_dnn_file_vec_layer1=[N_2_test_file(qt_dnn_baseDir,N,C,qt_dnn_layer1,qt_dnn_num_epochs,qt_dnn_num_suffix) for N in qt_dnn_N_vec]
qt_dnn_std_loss_vec_layer1=[]
qt_dnn_num_params_vec_layer1=[]

for file in qt_dnn_file_vec_layer1:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_dnn_std_loss_vec_layer1.append(std_loss)
    qt_dnn_num_params_vec_layer1.append(num_params)

qt_dnn_std_loss_vec_layer1=np.array(qt_dnn_std_loss_vec_layer1)
qt_dnn_relative_error_layer1=(qt_dnn_std_loss_vec_layer1/abs_avg_Y_train_vec)

#qt_dnn layer 2
qt_dnn_layer2=qt_dnn_layer_vec[1]
qt_dnn_file_vec_layer2=[N_2_test_file(qt_dnn_baseDir,N,C,qt_dnn_layer2,qt_dnn_num_epochs,qt_dnn_num_suffix) for N in qt_dnn_N_vec]

qt_dnn_std_loss_vec_layer2=[]
qt_dnn_num_params_vec_layer2=[]
for file in qt_dnn_file_vec_layer2:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_dnn_std_loss_vec_layer2.append(std_loss)
    qt_dnn_num_params_vec_layer2.append(num_params)

qt_dnn_std_loss_vec_layer2=np.array(qt_dnn_std_loss_vec_layer2)
qt_dnn_relative_error_layer2=(qt_dnn_std_loss_vec_layer2/abs_avg_Y_train_vec)


#qt_dnn layer 3
qt_dnn_layer3=qt_dnn_layer_vec[2]
qt_dnn_file_vec_layer3=[N_2_test_file(qt_dnn_baseDir,N,C,qt_dnn_layer3,qt_dnn_num_epochs,qt_dnn_num_suffix) for N in qt_dnn_N_vec]
qt_dnn_std_loss_vec_layer3=[]
qt_dnn_num_params_vec_layer3=[]
for file in qt_dnn_file_vec_layer3:
    std_loss = file_2_std(file)
    num_params = file_2_num_params(file)
    qt_dnn_std_loss_vec_layer3.append(std_loss)
    qt_dnn_num_params_vec_layer3.append(num_params)

qt_dnn_std_loss_vec_layer3=np.array(qt_dnn_std_loss_vec_layer3)
qt_dnn_relative_error_layer3=(qt_dnn_std_loss_vec_layer3/abs_avg_Y_train_vec)


# =============================================================================
# Generate CSV tables for Relative Error per layer across Lattice Sizes
# =============================================================================

# Layer 1
df_layer1 = pd.DataFrame({
    "Lattice_Size_N": pd.Series(qt_efnn_N_vec),
    "EFNN_relative_error": pd.Series(qt_efnn_relative_error_layer0),
    "DenseNet_relative_error": pd.Series(qt_densenet_relative_error_layer1),
    "ResNet_relative_error": pd.Series(qt_resnet_relative_error_layer1),
    "DNN_relative_error": pd.Series(qt_dnn_relative_error_layer1)
})
df_layer1.to_csv(f"./qt_relative_error_larger_lattice_layer1_C{C}.csv", index=False)
print(f"Saved qt_relative_error_larger_lattice_layer1_C{C}.csv")

# Layer 2
df_layer2 = pd.DataFrame({
    "Lattice_Size_N": pd.Series(qt_efnn_N_vec),
    "EFNN_relative_error": pd.Series(qt_efnn_relative_error_layer1),
    "DenseNet_relative_error": pd.Series(qt_densenet_relative_error_layer2),
    "ResNet_relative_error": pd.Series(qt_resnet_relative_error_layer2),
    "DNN_relative_error": pd.Series(qt_dnn_relative_error_layer2)
})
df_layer2.to_csv(f"./qt_relative_error_larger_lattice_layer2_C{C}.csv", index=False)
print(f"Saved qt_relative_error_larger_lattice_layer2_C{C}.csv")

# Layer 3
df_layer3 = pd.DataFrame({
    "Lattice_Size_N": pd.Series(qt_efnn_N_vec),
    "EFNN_relative_error": pd.Series(qt_efnn_relative_error_vec_layer2), # Note: script uses _vec_layer2 here
    "DenseNet_relative_error": pd.Series(qt_densenet_relative_error_layer3),
    "ResNet_relative_error": pd.Series(qt_resnet_relative_error_layer3),
    "DNN_relative_error": pd.Series(qt_dnn_relative_error_layer3)
})
df_layer3.to_csv(f"./qt_relative_error_larger_lattice_layer3_C{C}.csv", index=False)
print(f"Saved qt_relative_error_larger_lattice_layer3_C{C}.csv")