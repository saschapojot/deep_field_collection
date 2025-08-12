import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pickle
from pathlib import Path

#this script compares test loss for the same N, same epoch, different layer numbers, different C values
#this script needs to manually input lin's mse, from slurm output on supercomputer

N=10
mpl.rcParams['axes.linewidth'] = 2.5  # Set for all plots
C_vec=[1,3,5,7,9]
layer_vec=[1,2,3]
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


def std_loss_all_one_epoch(epoch_num,layer,N,C_vec):
    ret_std_loss_vec=[]
    for C in C_vec:
        oneFile=f"./out_model_data/N{N}/C{C}/layer{layer}/test_over_epochs.txt"
        ep, stdTmp = match_line_in_file(oneFile, epoch_num)
        print(f"ep={ep},C={C},layer={layer},sdTmp={stdTmp}")
        ret_std_loss_vec.append(stdTmp)
    return np.array(ret_std_loss_vec)


inDir=f"./train_test_data/N{N}/"
in_pkl_train_file=inDir+"/db.train_num_samples200000.pkl"
with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)
Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

Y_train_avg=np.mean(Y_train_array)
abs_Y_train_avg=np.abs(Y_train_avg)
print(f"Y_train_array.shape={Y_train_array.shape}")
set_epoch=675

layer1=layer_vec[0]
std_for_layer1=std_loss_all_one_epoch(set_epoch,layer1,N,C_vec)

layer2=layer_vec[1]
std_for_layer2=std_loss_all_one_epoch(set_epoch,layer2,N,C_vec)

layer3=layer_vec[2]
std_for_layer3=std_loss_all_one_epoch(set_epoch,layer3,N,C_vec)

relative_acc_layer1=std_for_layer1/abs_Y_train_avg
relative_acc_layer2=std_for_layer2/abs_Y_train_avg
relative_acc_layer3=std_for_layer3/abs_Y_train_avg

print(f"set_epoch={set_epoch}, relative_acc_layer1={relative_acc_layer1}")
print(f"set_epoch={set_epoch}, relative_acc_layer2={relative_acc_layer2}")
print(f"set_epoch={set_epoch}, relative_acc_layer3={relative_acc_layer3}")


out_pic_dir="./out_model_data/"
Path(out_pic_dir).mkdir(parents=True, exist_ok=True)
width=6
height=8
textSize=33
yTickSize=33
xTickSize=33
legend_fontsize=20
lineWidth1=3
marker_size1=100
tick_length=13
tick_width=2
minor_tick_length=7
minor_tick_width=1

plt.figure(figsize=(width, height))
plt.minorticks_on()

plt.scatter(C_vec,relative_acc_layer1,color="blue",marker="o",s=marker_size1,label=f"resnet, n={layer1}")
plt.plot(C_vec,relative_acc_layer1,color="blue",linestyle="dashed",linewidth=lineWidth1)


plt.scatter(C_vec,relative_acc_layer2,color="magenta",marker="^",s=marker_size1,label=f"resnet, n={layer2}")
plt.plot(C_vec,relative_acc_layer2,color="magenta",linestyle="dashed",linewidth=lineWidth1)

plt.scatter(C_vec,relative_acc_layer3,color="green",marker="s",s=marker_size1,label=f"resnet, n={layer3}")
plt.plot(C_vec,relative_acc_layer3,color="green",linestyle="dashed",linewidth=lineWidth1)

lin_mean_mse=0.9847254886017068
lin_mean_std=np.sqrt(lin_mean_mse)
lin_err_relative=lin_mean_std/abs_Y_train_avg

# plt.axhline(y=lin_err_relative, color="black", linestyle="--", label=f"Effective model",linewidth=lineWidth1)
plt.xlabel("$C$",fontsize=textSize)
plt.ylabel("Relative error",fontsize=textSize)

plt.legend(loc="upper right", bbox_to_anchor=(0.95, 0.8), fontsize=legend_fontsize)
ax = plt.gca()
# Format y-axis
formatter = plt.ScalarFormatter(useOffset=False, useMathText=True)
formatter.set_powerlimits((-3, -3))  # Force 10^{-3} scale
ax.yaxis.set_major_formatter(formatter)

plt.gca().yaxis.get_offset_text().set_fontsize(yTickSize)  # Set the size of the exponent

plt.tight_layout()
plt.savefig(out_pic_dir+f"epoch_{set_epoch}_N{N}_pbc.svg",bbox_inches='tight')
plt.savefig(out_pic_dir+f"epoch_{set_epoch}_N{N}_pbc.png",bbox_inches='tight')