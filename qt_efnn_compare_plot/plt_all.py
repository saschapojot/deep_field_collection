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
    test_txt_file = in_file_dir + f"/test_over{50}_rate0.9_epoch{num_epochs}_num_samples{num_suffix}.txt"
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
qt_densenet_baseDir="/home/adada/Documents/pyCode/deep_field_collection/qt_densenet_pbc/"
qt_densenet_layer_vec=np.array([2,3,4])
qt_densenet_N_vec=np.array([10,15,20,25,30,35,40])
qt_densenet_num_suffix=40000
qt_densenet_num_epochs = 500

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
qt_resnet_baseDir="/home/adada/Documents/pyCode/deep_field_collection/qt_resnet_pbc/"

qt_resnet_layer_vec=np.array([1,2,3])
qt_resnet_N_vec=np.array([10,15,20,25,30,35,40])
qt_resnet_num_suffix=40000
qt_resnet_num_epochs = 500

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
qt_dnn_baseDir="/home/adada/Documents/pyCode/deep_field_collection/qt_dnn/"
qt_dnn_layer_vec=np.array([1,2,3])
qt_dnn_N_vec=np.array([10,15,20,25,30,35,40])
qt_dnn_num_suffix=40000
qt_dnn_num_epochs = 500

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


#######
width=6
height=8
textSize=33
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

#qt_efnn
plt.figure(figsize=(width, height))
plt.minorticks_on()

#qt_efnn layer 1
plt.scatter(qt_efnn_N_vec, qt_efnn_relative_error_layer0, color="green", label="EFNN, 1 layer",s=marker_size2)
plt.plot(qt_efnn_N_vec, qt_efnn_relative_error_layer0, color="green", linestyle="dashed",linewidth=lineWidth2)
#qt_efnn layer 2
plt.scatter(qt_efnn_N_vec, qt_efnn_relative_error_layer1, color="darkgreen", label="EFNN, 2 layers",s=marker_size2)
plt.plot(qt_efnn_N_vec, qt_efnn_relative_error_layer1, color="darkgreen", linestyle="dashed",linewidth=lineWidth2)
#qt_efnn layer 3
plt.scatter(qt_efnn_N_vec, qt_efnn_relative_error_vec_layer2, color="limegreen", label="EFNN, 3 layers",s=marker_size2)
plt.plot(qt_efnn_N_vec, qt_efnn_relative_error_vec_layer2, color="limegreen", linestyle="dashed",linewidth=lineWidth2)


#qt_densenet
#qt_densenet layer 1
plt.scatter(qt_densenet_N_vec, qt_densenet_relative_error_layer1,marker="s", color="red", label="DenseNet, 1 layer",s=marker_size2)
plt.plot(qt_densenet_N_vec, qt_densenet_relative_error_layer1, color="red", linestyle="dashed", linewidth=lineWidth2)
#qt_densenet layer 2
plt.scatter(qt_densenet_N_vec, qt_densenet_relative_error_layer2,marker="s", color="darkred", label="DenseNet, 2 layers",s=marker_size2)
plt.plot(qt_densenet_N_vec, qt_densenet_relative_error_layer2, color="darkred", linestyle="dashed", linewidth=lineWidth2)
#qt_densenet layer 3
plt.scatter(qt_densenet_N_vec, qt_densenet_relative_error_layer3,marker="s", color="indianred", label="DenseNet, 3 layers",s=marker_size2)
plt.plot(qt_densenet_N_vec, qt_densenet_relative_error_layer3, color="indianred", linestyle="dashed", linewidth=lineWidth2)

#qt_resnet
#qt_resnet layer 1
plt.scatter(qt_resnet_N_vec, qt_resnet_relative_error_layer1,marker="^", color="purple", label="ResNet, 1 layer", s=marker_size2)
plt.plot(qt_resnet_N_vec, qt_resnet_relative_error_layer1, color="purple", linestyle="dashed", linewidth=lineWidth2)
#qt_resnet layer 2
plt.scatter(qt_resnet_N_vec, qt_resnet_relative_error_layer2,marker="^", color="darkmagenta", label="ResNet, 2 layers", s=marker_size2)
plt.plot(qt_resnet_N_vec, qt_resnet_relative_error_layer2, color="darkmagenta", linestyle="dashed", linewidth=lineWidth2)
#qt_resnet layer 3
plt.scatter(qt_resnet_N_vec, qt_resnet_relative_error_layer3,marker="^", color="mediumorchid", label="ResNet, 3 layers", s=marker_size2)
plt.plot(qt_resnet_N_vec, qt_resnet_relative_error_layer3, color="mediumorchid", linestyle="dashed", linewidth=lineWidth2)

#qt_dnn layer1
plt.scatter(qt_dnn_N_vec,qt_dnn_relative_error_layer1,marker="D", color="blue", label="DNN, 1 layer", s=marker_size2)
plt.plot(qt_dnn_N_vec,qt_dnn_relative_error_layer1,color="blue", linestyle="dashed")

#qt_dnn layer 2
plt.scatter(qt_dnn_N_vec,qt_dnn_relative_error_layer2,marker="D", color="darkblue", label="DNN, 2 layers", s=marker_size2)
plt.plot(qt_dnn_N_vec,qt_dnn_relative_error_layer2,color="darkblue", linestyle="dashed")

#qt_dnn layer 3
plt.scatter(qt_dnn_N_vec,qt_dnn_relative_error_layer3,marker="D", color="cornflowerblue", label="DNN, 3 layers", s=marker_size2)
plt.plot(qt_dnn_N_vec,qt_dnn_relative_error_layer3,color="cornflowerblue", linestyle="dashed")


plt.axhline(y=lin_err_relative, color="black", linestyle="--", label=f"Effective model",linewidth=lineWidth1)

plt.xlabel('Lattice Size (N)',fontsize=textSize)
plt.ylabel('Relative Error',fontsize=textSize)
# plt.title('Quantum double exchange model')
plt.tick_params(axis='both', length=tick_length, width=tick_width)
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width)
plt.xticks([10,20,30,40],labels=["10","20","30","40"],fontsize=xTickSize)
plt.yticks(fontsize=yTickSize)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
# Get the legend object
legend = plt.legend(loc="upper right", fontsize=legend_fontsize-10,
          ncol=2,
          handlelength=0.8,
          handletextpad=0.2,
          columnspacing=0.3,
          borderpad=0.1,
          labelspacing=0.15,
          markerscale=0.6)

# Make the legend more transparent
legend.get_frame().set_alpha(0.7)  # Adjust alpha value (0=fully transparent, 1=opaque)
plt.yscale('log')  # Consider log scale if ranges vary widely
plt.xticks()
plt.yticks(fontsize=yTickSize)
plt.subplots_adjust(left=0.3, right=0.95, top=0.99, bottom=0.15)
plt.savefig('qt_efnn_vs_all', dpi=300)
plt.savefig('qt_efnn_vs_all.svg')

#make csv
# Add parameter counts to a separate DataFrame
params_data = {
    'Lattice_Size': qt_efnn_N_vec,
    'EFNN_1layer': qt_efnn_num_params_vec_layer0,
    'EFNN_2layers': qt_efnn_num_params_vec_layer1,
    'EFNN_3layers': qt_efnn_num_params_vec_layer2,
    'DenseNet_1layer': qt_densenet_num_params_vec_layer1,
    'DenseNet_2layers': qt_densenet_num_params_vec_layer2,
    'DenseNet_3layers': qt_densenet_num_params_vec_layer3,
    'ResNet_1layer': qt_resnet_num_params_vec_layer1,
    'ResNet_2layers': qt_resnet_num_params_vec_layer2,
    'ResNet_3layers': qt_resnet_num_params_vec_layer3,
    "DNN_1layer":qt_dnn_num_params_vec_layer1,
    "DNN_2layers": qt_dnn_num_params_vec_layer2,
    "DNN_3layers": qt_dnn_num_params_vec_layer3,
}
params_df = pd.DataFrame(params_data)
params_df.to_csv('quantum_nn_parameter_counts.csv', index=False)