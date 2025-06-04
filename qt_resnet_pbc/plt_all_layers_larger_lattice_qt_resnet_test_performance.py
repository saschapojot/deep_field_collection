import re
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.linear_model import LinearRegression
import pickle
import matplotlib as mpl
from pathlib import Path
mpl.rcParams['axes.linewidth'] = 2.5  # Set for all plots
layer_vec=np.array([1,2,3])
C=20
N_vec=np.array([10,15,20,25,30,35,40])
num_suffix=40000
num_epochs = 500#optimal is ?
inDirRoot="./larger_lattice_test_performance/"

def N_2_test_file(N,layer_num):
    in_file_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file = in_file_dir + f"/test_epoch{num_epochs}_num_samples{num_suffix}.txt"
    return test_txt_file


def N_2_test_data_pkl(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/"
    pkl_test_file=in_model_dir+f"/db.test_num_samples{num_suffix}.pkl"
    return pkl_test_file


pattern_std=r'std_loss=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
pattern_custom_err=r'custom_err\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'

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


pkl_file_vec=[N_2_test_data_pkl(N) for N in N_vec]

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

#layer1
layer1=layer_vec[0]
file_vec_layer1=[N_2_test_file(N,layer1) for N in N_vec]
std_loss_vec_layer1=[]
for file in file_vec_layer1:
    std_loss=file_2_std(file)
    std_loss_vec_layer1.append(std_loss)

std_loss_vec_layer1=np.array(std_loss_vec_layer1)
relative_error_layer1=(std_loss_vec_layer1/abs_avg_Y_train_vec)

# layer2
layer2=layer_vec[1]
file_vec_layer2=[N_2_test_file(N,layer2) for N in N_vec]
std_loss_vec_layer2=[]
for file in file_vec_layer2:
    std_loss=file_2_std(file)
    std_loss_vec_layer2.append(std_loss)

std_loss_vec_layer2=np.array(std_loss_vec_layer2)

relative_error_layer2=(std_loss_vec_layer2/abs_avg_Y_train_vec)

#layer 3
layer3=layer_vec[2]
file_vec_layer3=[N_2_test_file(N,layer3) for N in N_vec]
std_loss_vec_layer3=[]

for file in file_vec_layer3:
    std_loss = file_2_std(file)
    std_loss_vec_layer3.append(std_loss)

std_loss_vec_layer3=np.array(std_loss_vec_layer3)
relative_error_layer3=(std_loss_vec_layer3/abs_avg_Y_train_vec)


width=6
height=8
textSize=33
yTickSize=33
xTickSize=33
legend_fontsize=24
lineWidth1=3
marker_size1=100
tick_length=13
tick_width=2
minor_tick_length=7
minor_tick_width=1

out_db_qt_dir="./larger_lattice_test_performance/"
Path(out_db_qt_dir).mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(width, height))

plt.minorticks_on()

#layer1
plt.scatter(N_vec,relative_error_layer1,color="blue",marker="o",s=marker_size1,label=f"resnet, n={layer1}")
plt.plot(N_vec,relative_error_layer1,color="blue",linestyle="dashed",linewidth=lineWidth1)


#layer 2
plt.scatter(N_vec,relative_error_layer2,color="magenta",marker="^",s=marker_size1,label=f"resnet, n={layer2}")
plt.plot(N_vec,relative_error_layer2,color="magenta",linestyle="dashed",linewidth=lineWidth1)


#layer 2
plt.scatter(N_vec,relative_error_layer3,color="green",marker="s",s=marker_size1,label=f"resnet, n={layer3}")
plt.plot(N_vec,relative_error_layer3,color="green",linestyle="dashed",linewidth=lineWidth1)
plt.legend(loc="upper right", bbox_to_anchor=(0.9, 0.8), fontsize=legend_fontsize)
plt.xlabel("$N$", fontsize=textSize)
plt.ylabel("Relative error",fontsize=textSize)
plt.xticks([10, 20, 30, 40], ["10", "20", "30", "40"], fontsize=xTickSize)

# plt.yticks([3e-3,2e-3,1e-3],["3","2","1"], fontsize=yTickSize)

plt.savefig(out_db_qt_dir + f"/relative_error_C{C}_epoch{num_epochs}_pbc.png", bbox_inches="tight")  # bbox_inches ensures no truncation

plt.savefig(out_db_qt_dir + f"/relative_error_C{C}_epoch{num_epochs}_pbc.svg", bbox_inches="tight")  # bbox_inches ensures no truncation
plt.close()