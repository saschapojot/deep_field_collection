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
#all layers

C_vec=[10,15,20,25]
N=10#plots for N=10
set_epoch=500
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
        # print(f"C={C}, oneFile={oneFile}")
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
qt_efnn_num_epochs = set_epoch

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
qt_densenet_num_epochs = set_epoch

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
qt_resnet_num_epochs = set_epoch

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
qt_dnn_num_epochs = set_epoch

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
line1=plt.scatter(C_vec,qt_efnn_relative_err_layer0,color="green", label="EFNN, 1 layer",s=marker_size2)
plt.plot(C_vec,qt_efnn_relative_err_layer0,color="green", linestyle="dashed",linewidth=lineWidth2)

#qt_efnn layer 2
line2=plt.scatter(C_vec,qt_efnn_relative_err_layer1,color="darkgreen", label="EFNN, 2 layers",s=marker_size2)
plt.plot(C_vec,qt_efnn_relative_err_layer1,color="darkgreen", linestyle="dashed", linewidth=lineWidth2)


#qt_efnn layer 3
line3=plt.scatter(C_vec,qt_efnn_relative_err_layer2,color="limegreen", label="EFNN, 3 layers",s=marker_size2)
plt.plot(C_vec,qt_efnn_relative_err_layer2,color="limegreen", linestyle="dashed", linewidth=lineWidth2)


#qt_densenet layer 1
line4=plt.scatter(C_vec,qt_densenet_relative_err_layer0,marker="s", color="red", label="DenseNet, 1 layer",s=marker_size2)
plt.plot(C_vec,qt_densenet_relative_err_layer0, color="red", linestyle="dashed", linewidth=lineWidth2)


#qt_densenet layer 2
line5=plt.scatter(C_vec,qt_densenet_relative_err_layer1,marker="s", color="darkred", label="DenseNet, 2 layers", s=marker_size2)
plt.plot(C_vec,qt_densenet_relative_err_layer1,  color="darkred", linestyle="dashed", linewidth=lineWidth2)


#qt_densenet layer 3
line6=plt.scatter(C_vec,qt_densenet_relative_err_layer2,marker="s", color="indianred", label="DenseNet, 3 layers", s=marker_size2)
plt.plot(C_vec,qt_densenet_relative_err_layer2,  color="indianred", linestyle="dashed", linewidth=lineWidth2)


#qt_resnet layer 1
line7=plt.scatter(C_vec,qt_resnet_relative_err_layer0,marker="^", color="purple", label="ResNet, 1 layer", s=marker_size2)
plt.plot(C_vec,qt_resnet_relative_err_layer0,color="purple", linestyle="dashed", linewidth=lineWidth2)

#qt_resnet layer 2
line8=plt.scatter(C_vec,qt_resnet_relative_err_layer1,marker="^", color="darkmagenta", label="ResNet, 2 layers", s=marker_size2)
plt.plot(C_vec,qt_resnet_relative_err_layer1,color="darkmagenta", linestyle="dashed", linewidth=lineWidth2)

#qt_resnet layer 3
line9=plt.scatter(C_vec,qt_resnet_relative_err_layer2,marker="^", color="mediumorchid", label="ResNet, 3 layers", s=marker_size2)
plt.plot(C_vec,qt_resnet_relative_err_layer2,color="mediumorchid", linestyle="dashed", linewidth=lineWidth2)

#qt_dnn layer1
line10 = plt.scatter(C_vec,qt_dnn_relative_err_layer0,marker="D", color="blue", label="DNN, 1 layer", s=marker_size2)
plt.plot(C_vec,qt_dnn_relative_err_layer0,color="blue", linestyle="dashed",linewidth=lineWidth2)


#qt_dnn layer2
line11 = plt.scatter(C_vec,qt_dnn_relative_err_layer1,marker="D", color="darkblue", label="DNN, 2 layers", s=marker_size2)
plt.plot(C_vec,qt_dnn_relative_err_layer1,color="darkblue", linestyle="dashed", linewidth=lineWidth2)

#qt_dnn layer3
line12 = plt.scatter(C_vec,qt_dnn_relative_err_layer2,marker="D", color="cornflowerblue", label="DNN, 3 layers", s=marker_size2)
plt.plot(C_vec,qt_dnn_relative_err_layer2,color="cornflowerblue", linestyle="dashed", linewidth=lineWidth2)




lin_mean_mse=0.9847254886017068
lin_mean_std=np.sqrt(lin_mean_mse)
lin_err_relative=lin_mean_std/abs_Y_train_avg
plt.axhline(y=lin_err_relative, color="black", linestyle="--", label=f"Effective model",linewidth=lineWidth1)

plt.xlabel('$C$',fontsize=textSize)
plt.ylabel('Relative Error',fontsize=textSize)

plt.tick_params(axis='both', length=tick_length, width=tick_width)
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width)

plt.xticks([10,15,20,25],labels=["10","15","20","25"],fontsize=xTickSize)
plt.yticks(fontsize=yTickSize)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
# legend = plt.legend(loc="best", fontsize=legend_fontsize-10,)

plt.yscale('log')  # Consider log scale if ranges vary widely

# Format y-axis to show values multiplied by 10^3
from matplotlib.ticker import FuncFormatter
def format_func(value, tick_number):
    return f'{value*1e3:.1f}'

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

# Add the scale text inside the plot area
plt.text(0.02, 1.02, r'$\times 10^{-3}$', transform=plt.gca().transAxes,
         fontsize=textSize, verticalalignment='bottom')
plt.subplots_adjust(left=0.3, right=0.95, top=0.93, bottom=0.15)
plt.savefig(f'epoch{set_epoch}_all_layers_qt_efnn_vs_all_C.png', dpi=300)
plt.savefig(f'epoch{set_epoch}_all_layers_qt_efnn_vs_all_C.svg')
plt.close()

# Create a figure for vertical legend
fig_legend = plt.figure(figsize=(4, 8))  # Taller than wide

# Create dummy lines for the legend
legend_elements = [
    plt.Line2D([0], [0], color='green', marker='o', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='EFNN, 1 layer'),
    plt.Line2D([0], [0], color='darkgreen', marker='o', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='EFNN, 2 layers'),
    plt.Line2D([0], [0], color='limegreen', marker='o', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='EFNN, 3 layers'),
    plt.Line2D([0], [0], color='red', marker='s', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='DenseNet, 1 layer'),
    plt.Line2D([0], [0], color='darkred', marker='s', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='DenseNet, 2 layers'),
    plt.Line2D([0], [0], color='indianred', marker='s', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='DenseNet, 3 layers'),
    plt.Line2D([0], [0], color='purple', marker='^', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='ResNet, 1 layer'),
    plt.Line2D([0], [0], color='darkmagenta', marker='^', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='ResNet, 2 layers'),
    plt.Line2D([0], [0], color='mediumorchid', marker='^', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='ResNet, 3 layers'),
    plt.Line2D([0], [0], color='blue', marker='D', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='DNN, 1 layer'),
    plt.Line2D([0], [0], color='darkblue', marker='D', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='DNN, 2 layers'),
    plt.Line2D([0], [0], color='cornflowerblue', marker='D', linestyle='dashed', markersize=10, linewidth=lineWidth2, label='DNN, 3 layers'),
    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=lineWidth1, label='Effective model')
]

# Create an axes that fills the entire figure
ax = fig_legend.add_subplot(111)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Create the legend positioned to fill the axes
legend = ax.legend(handles=legend_elements,
                   loc='center',
                   bbox_to_anchor=(0.5, 0.5),
                   ncol=1,  # Single column
                   fontsize=legend_fontsize-5,
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   borderaxespad=0,
                   columnspacing=1.0,
                   handlelength=2.5,  # Adjust length of legend lines
                   handletextpad=0.8)  # Adjust space between line and text

# Make everything invisible except the legend
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_facecolor('white')
fig_legend.patch.set_facecolor('white')

# Get the exact size of the legend
fig_legend.canvas.draw()
bbox = legend.get_window_extent()
bbox = bbox.transformed(fig_legend.dpi_scale_trans.inverted())
width, height = bbox.width, bbox.height

# Resize the figure to match the legend size
fig_legend.set_size_inches(width + 0.1, height + 0.1)  # Add small padding

# Save with minimal padding
fig_legend.savefig(f'epoch{set_epoch}_all_layers_qt_efnn_vs_all_C_legend_vertical.png',
                   dpi=300,
                   bbox_inches='tight',
                   pad_inches=0.02,
                   facecolor='white',
                   edgecolor='none')
fig_legend.savefig(f'epoch{set_epoch}_all_layers_qt_efnn_vs_all_C_legend_vertical.svg',
                   bbox_inches='tight',
                   pad_inches=0.02,
                   facecolor='white',
                   edgecolor='none')
plt.close(fig_legend)