import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from itertools import combinations
import matplotlib as mpl
#this script plots test error for
#hs_efnn, hs_attn, hs_densenet , hs_longer_attn, hs_resnet

L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)
# Set the color explicitly to black
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 2.5  # You already have this

#load result from hs_efnn
hs_efnn_inPath="/home/adada/Documents/pyCode/deep_field_collection/hs_efnn/compare_layer_neuron_num/"

hs_efnn_epoch_num=15999
hs_efnn_layer_num_vec=[1,2,3]

hs_efnn_inCsvName_layer0=hs_efnn_inPath+f"/layer{hs_efnn_layer_num_vec[0]}_epoch{hs_efnn_epoch_num}_std_loss.csv"
hs_efnn_inCsvName_layer1=hs_efnn_inPath+f"/layer{hs_efnn_layer_num_vec[1]}_epoch{hs_efnn_epoch_num}_std_loss.csv"
hs_efnn_inCsvName_layer2=hs_efnn_inPath+f"/layer{hs_efnn_layer_num_vec[2]}_epoch{hs_efnn_epoch_num}_std_loss.csv"

hs_efnn_in_df_layer0=pd.read_csv(hs_efnn_inCsvName_layer0)
hs_efnn_in_df_layer1=pd.read_csv(hs_efnn_inCsvName_layer1)
hs_efnn_in_df_layer2=pd.read_csv(hs_efnn_inCsvName_layer2)

#hs_efnn , layer 1
hs_efnn_num_neurons_vec_layer0=np.array(hs_efnn_in_df_layer0["num_neurons"])
hs_efnn_std_loss_vec_layer0=np.array(hs_efnn_in_df_layer0["std_loss"])
hs_efnn_num_parameters_vec_layer0=np.array(hs_efnn_in_df_layer0["num_params"])
# print(f"hs_efnn_num_parameters_vec_layer0={hs_efnn_num_parameters_vec_layer0}")
#hs_efnn , layer 2
hs_efnn_num_neurons_vec_layer1=np.array(hs_efnn_in_df_layer1["num_neurons"])
hs_efnn_std_loss_vec_layer1=np.array(hs_efnn_in_df_layer1["std_loss"])
hs_efnn_num_parameters_vec_layer1=np.array(hs_efnn_in_df_layer1["num_params"])
#hs_efnn , layer 3
hs_efnn_num_neurons_vec_layer2=np.array(hs_efnn_in_df_layer2["num_neurons"])
hs_efnn_std_loss_vec_layer2=np.array(hs_efnn_in_df_layer2["std_loss"])
hs_efnn_num_parameters_vec_layer2=np.array(hs_efnn_in_df_layer2["num_params"])

# #load result from hs_attn
# hs_attn_inPath="/home/adada/Documents/pyCode/deep_field_collection/hs_attn/compare_layer_neuron_num/"
# hs_attn_epoch_num=15999
# hs_attn_layer_num_vec=[1,2,3]
#
# hs_attn_inCsvName_layer0=hs_attn_inPath+f"/layer{hs_attn_layer_num_vec[0]}_epoch{hs_attn_epoch_num}_std_loss.csv"
# hs_attn_inCsvName_layer1=hs_attn_inPath+f"/layer{hs_attn_layer_num_vec[1]}_epoch{hs_attn_epoch_num}_std_loss.csv"
# hs_attn_inCsvName_layer2=hs_attn_inPath+f"/layer{hs_attn_layer_num_vec[2]}_epoch{hs_attn_epoch_num}_std_loss.csv"
#
# hs_attn_in_df_layer0=pd.read_csv(hs_attn_inCsvName_layer0)
# hs_attn_in_df_layer1=pd.read_csv(hs_attn_inCsvName_layer1)
# hs_attn_in_df_layer2=pd.read_csv(hs_attn_inCsvName_layer2)



#load result from hs_densenet
hs_densenet_inPath="/home/adada/Documents/pyCode/deep_field_collection/hs_densenet/compare_layer_neuron_num/"

hs_densenet_epoch_num=15999
hs_densenet_layer_num_vec=[2,3,4]

hs_densenet_inCsvName_layer0=hs_densenet_inPath+f"/layer{hs_densenet_layer_num_vec[0]}_epoch{hs_densenet_epoch_num}_std_loss.csv"
hs_densenet_inCsvName_layer1=hs_densenet_inPath+f"/layer{hs_densenet_layer_num_vec[1]}_epoch{hs_densenet_epoch_num}_std_loss.csv"
hs_densenet_inCsvName_layer2=hs_densenet_inPath+f"/layer{hs_densenet_layer_num_vec[2]}_epoch{hs_densenet_epoch_num}_std_loss.csv"



hs_densenet_in_df_layer0=pd.read_csv(hs_densenet_inCsvName_layer0)
hs_densenet_in_df_layer1=pd.read_csv(hs_densenet_inCsvName_layer1)
hs_densenet_in_df_layer2=pd.read_csv(hs_densenet_inCsvName_layer2)

#hs_densenet , layer 1
hs_densenet_growth_rate_vec_layer0=np.array(hs_densenet_in_df_layer0["growth_rate"])
hs_densenet_std_loss_vec_layer0=np.array(hs_densenet_in_df_layer0["std_loss"])
hs_densenet_num_parameters_vec_layer0=np.array(hs_densenet_in_df_layer0["num_params"])
#hs_densenet , layer 2
hs_densenet_growth_rate_vec_layer1=np.array(hs_densenet_in_df_layer1["growth_rate"])
hs_densenet_std_loss_vec_layer1=np.array(hs_densenet_in_df_layer1["std_loss"])
hs_densenet_num_parameters_vec_layer1=np.array(hs_densenet_in_df_layer1["num_params"])
#hs_densenet , layer 3
hs_densenet_growth_rate_vec_layer2=np.array(hs_densenet_in_df_layer2["growth_rate"])
hs_densenet_std_loss_vec_layer2=np.array(hs_densenet_in_df_layer2["std_loss"])
hs_densenet_num_parameters_vec_layer2=np.array(hs_densenet_in_df_layer2["num_params"])

#load result from hs_resnet
#hs_resnet , layer 1
hs_resnet_inPath="/home/adada/Documents/pyCode/deep_field_collection/hs_resnet/compare_layer_neuron_num/"

hs_resnet_epoch_num=15999
hs_resnet_layer_num_vec=[1,2,3]

hs_resnet_inCsvName_layer0=hs_resnet_inPath+f"/layer{hs_resnet_layer_num_vec[0]}_epoch{hs_resnet_epoch_num}_std_loss.csv"
hs_resnet_inCsvName_layer1=hs_resnet_inPath+f"/layer{hs_resnet_layer_num_vec[1]}_epoch{hs_resnet_epoch_num}_std_loss.csv"
hs_resnet_inCsvName_layer2=hs_resnet_inPath+f"/layer{hs_resnet_layer_num_vec[2]}_epoch{hs_resnet_epoch_num}_std_loss.csv"


hs_resnet_in_df_layer0=pd.read_csv(hs_resnet_inCsvName_layer0)
hs_resnet_in_df_layer1=pd.read_csv(hs_resnet_inCsvName_layer1)
hs_resnet_in_df_layer2=pd.read_csv(hs_resnet_inCsvName_layer2)

#hs_resnet , layer 1
hs_resnet_num_neurons_vec_layer0=np.array(hs_resnet_in_df_layer0["num_neurons"])
hs_resnet_std_loss_vec_layer0=np.array(hs_resnet_in_df_layer0["std_loss"])
hs_resnet_num_parameters_vec_layer0=np.array(hs_resnet_in_df_layer0["num_params"])
#hs_resnet , layer 2
hs_resnet_num_neurons_vec_layer1=np.array(hs_resnet_in_df_layer1["num_neurons"])
hs_resnet_std_loss_vec_layer1=np.array(hs_resnet_in_df_layer1["std_loss"])
hs_resnet_num_parameters_vec_layer1=np.array(hs_resnet_in_df_layer1["num_params"])
#hs_resnet , layer 3
hs_resnet_num_neurons_vec_layer2=np.array(hs_resnet_in_df_layer2["num_neurons"])
hs_resnet_std_loss_vec_layer2=np.array(hs_resnet_in_df_layer2["std_loss"])
hs_resnet_num_parameters_vec_layer2=np.array(hs_resnet_in_df_layer2["num_params"])

#load result from hs_dnn

hs_dnn_inPath=f"/home/adada/Documents/pyCode/deep_field_collection/hs_dnn/compare_layer_neuron_num/"
hs_dnn_epoch_num=15999
hs_dnn_layer_num_vec=[1,2,3]

hs_dnn_inCsvName_layer0=hs_dnn_inPath+f"/layer{hs_dnn_layer_num_vec[0]}_epoch{hs_dnn_epoch_num}_std_loss.csv"
hs_dnn_inCsvName_layer1=hs_dnn_inPath+f"/layer{hs_dnn_layer_num_vec[1]}_epoch{hs_dnn_epoch_num}_std_loss.csv"

hs_dnn_inCsvName_layer2=hs_dnn_inPath+f"/layer{hs_dnn_layer_num_vec[2]}_epoch{hs_dnn_epoch_num}_std_loss.csv"

hs_dnn_in_df_layer0=pd.read_csv(hs_dnn_inCsvName_layer0)
hs_dnn_in_df_layer1=pd.read_csv(hs_dnn_inCsvName_layer1)
hs_dnn_in_df_layer2=pd.read_csv(hs_dnn_inCsvName_layer2)

#hs_dnn, layer 1
hs_dnn_num_neurons_vec_layer0=np.array(hs_dnn_in_df_layer0["num_neurons"])
hs_dnn_std_loss_vec_layer0=np.array(hs_dnn_in_df_layer0["std_loss"])
hs_dnn_num_parameters_vec_layer0=np.array(hs_dnn_in_df_layer0["num_params"])

#hs_dnn, layer 2
hs_dnn_num_neurons_vec_layer1=np.array(hs_dnn_in_df_layer1["num_neurons"])
hs_dnn_std_loss_vec_layer1=np.array(hs_dnn_in_df_layer1["std_loss"])
hs_dnn_num_parameters_vec_layer1=np.array(hs_dnn_in_df_layer1["num_params"])

#hs_dnn, layer 3
hs_dnn_num_neurons_vec_layer2=np.array(hs_dnn_in_df_layer2["num_neurons"])
hs_dnn_std_loss_vec_layer2=np.array(hs_dnn_in_df_layer2["std_loss"])
hs_dnn_num_parameters_vec_layer2=np.array(hs_dnn_in_df_layer2["num_params"])
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

#efnn
plt.figure(figsize=(width, height))
plt.minorticks_on()
# hs_efnn , 1 layer
plt.scatter(hs_efnn_num_neurons_vec_layer0,hs_efnn_std_loss_vec_layer0,color="green",label="EFNN, 1 layer",s=marker_size2)
plt.plot(hs_efnn_num_neurons_vec_layer0,hs_efnn_std_loss_vec_layer0,color="green",linestyle="dashed", linewidth=lineWidth2)

# hs_efnn , 2 layers
plt.scatter(hs_efnn_num_neurons_vec_layer1, hs_efnn_std_loss_vec_layer1, color="darkgreen", label="EFNN, 2 layers",s=marker_size2)
plt.plot(hs_efnn_num_neurons_vec_layer1, hs_efnn_std_loss_vec_layer1, color="darkgreen", linestyle="dashed", linewidth=lineWidth2)

# hs_efnn , 3 layer2
plt.scatter(hs_efnn_num_neurons_vec_layer2, hs_efnn_std_loss_vec_layer2, color="limegreen", label="EFNN, 3 layers",s=marker_size2)
plt.plot(hs_efnn_num_neurons_vec_layer2, hs_efnn_std_loss_vec_layer2, color="limegreen", linestyle="dashed", linewidth=lineWidth2)



## densenet
# hs_densenet plots
plt.scatter(hs_densenet_growth_rate_vec_layer0, hs_densenet_std_loss_vec_layer0,marker="s", color="red", label="DenseNet, 1 layer", s=marker_size2)
plt.plot(hs_densenet_growth_rate_vec_layer0, hs_densenet_std_loss_vec_layer0, color="red", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(hs_densenet_growth_rate_vec_layer1, hs_densenet_std_loss_vec_layer1,marker="s", color="darkred", label="DenseNet, 2 layers", s=marker_size2)
plt.plot(hs_densenet_growth_rate_vec_layer1, hs_densenet_std_loss_vec_layer1, color="darkred", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(hs_densenet_growth_rate_vec_layer2, hs_densenet_std_loss_vec_layer2,marker="s", color="indianred", label="DenseNet, 3 layers", s=marker_size2)
plt.plot(hs_densenet_growth_rate_vec_layer2, hs_densenet_std_loss_vec_layer2, color="indianred", linestyle="dashed", linewidth=lineWidth2)


#resnet

# hs_resnet plots
plt.scatter(hs_resnet_num_neurons_vec_layer0, hs_resnet_std_loss_vec_layer0,marker="^", color="purple", label="ResNet, 1 layer",edgecolors='black', s=marker_size2)
plt.plot(hs_resnet_num_neurons_vec_layer0, hs_resnet_std_loss_vec_layer0, color="purple", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(hs_resnet_num_neurons_vec_layer1, hs_resnet_std_loss_vec_layer1,marker="^", color="darkmagenta", label="ResNet, 2 layers",edgecolors='black', s=marker_size2)
plt.plot(hs_resnet_num_neurons_vec_layer1, hs_resnet_std_loss_vec_layer1, color="darkmagenta", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(hs_resnet_num_neurons_vec_layer2, hs_resnet_std_loss_vec_layer2,marker="^", color="mediumorchid", label="ResNet, 3 layers",edgecolors='black', s=marker_size2)
plt.plot(hs_resnet_num_neurons_vec_layer2, hs_resnet_std_loss_vec_layer2, color="mediumorchid", linestyle="dashed", linewidth=lineWidth2)

#hs_dnn
# DNN plots
# hs_dnn, 1 layer
plt.scatter(hs_dnn_num_neurons_vec_layer0, hs_dnn_std_loss_vec_layer0, marker="D", color="blue", label="DNN, 1 layer", edgecolors='black', s=marker_size2)
plt.plot(hs_dnn_num_neurons_vec_layer0, hs_dnn_std_loss_vec_layer0, color="blue", linestyle="dashed", linewidth=lineWidth2)

# hs_dnn, 2 layers
plt.scatter(hs_dnn_num_neurons_vec_layer1, hs_dnn_std_loss_vec_layer1, marker="D", color="darkblue", label="DNN, 2 layers", edgecolors='black', s=marker_size2)
plt.plot(hs_dnn_num_neurons_vec_layer1, hs_dnn_std_loss_vec_layer1, color="darkblue", linestyle="dashed", linewidth=lineWidth2)

# hs_dnn, 3 layers
plt.scatter(hs_dnn_num_neurons_vec_layer2, hs_dnn_std_loss_vec_layer2, marker="D", color="cornflowerblue", label="DNN, 3 layers", edgecolors='black', s=marker_size2)
plt.plot(hs_dnn_num_neurons_vec_layer2, hs_dnn_std_loss_vec_layer2, color="cornflowerblue", linestyle="dashed", linewidth=lineWidth2)

plt.yscale("log")  # If your errors span multiple orders of magnitude
# Add labels, title and legend
plt.xlabel('Neuron Number',fontsize=textSize)
# plt.title(f'Heisenberg Spin System',fontsize=textSize)
plt.xticks([15,60,105,150],labels=["15","60","105","150"],fontsize=xTickSize)
plt.ylabel("Absolute error",fontsize=textSize)
plt.legend(loc="best", fontsize=legend_fontsize-10,  # Smaller font
          ncol=2,  # 2 columns instead of 3
          handlelength=0.8,      # Shorter marker lines
          handletextpad=0.2,     # Less space between marker and text
          columnspacing=0.3,     # Tighter column spacing
          borderpad=0.1,         # Less padding
          labelspacing=0.15,     # Less vertical space
          markerscale=0.6)       # Smaller markers
plt.yticks(fontsize=yTickSize)
plt.tick_params(axis='both', length=tick_length, width=tick_width)
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
# Improve layout and save
plt.subplots_adjust(left=0.3, right=0.95, top=0.99, bottom=0.15)

# plt.tight_layout()
plt.savefig('hs_efnn_vs_all.png', dpi=300)
plt.savefig('hs_efnn_vs_all.svg')
plt.close()


# # Plot number of parameters
plt.figure(figsize=(width, height))
plt.minorticks_on()

# EFNN 1 layer
plt.scatter(hs_efnn_num_neurons_vec_layer0, hs_efnn_num_parameters_vec_layer0, color="green",marker="o", label="EFNN, 1 layer", s=marker_size2)
plt.plot(hs_efnn_num_neurons_vec_layer0, hs_efnn_num_parameters_vec_layer0, color="green", linestyle="dashed", linewidth=lineWidth2)

# EFNN 2 layers
plt.scatter(hs_efnn_num_neurons_vec_layer1, hs_efnn_num_parameters_vec_layer1, color="darkgreen",marker="o", label="EFNN, 2 layers", s=marker_size2)
plt.plot(hs_efnn_num_neurons_vec_layer1, hs_efnn_num_parameters_vec_layer1, color="darkgreen", linestyle="dashed", linewidth=lineWidth2)

# EFNN 3 layers
plt.scatter(hs_efnn_num_neurons_vec_layer2, hs_efnn_num_parameters_vec_layer2, color="limegreen",marker="o", label="EFNN, 3 layers", s=marker_size2)
plt.plot(hs_efnn_num_neurons_vec_layer2, hs_efnn_num_parameters_vec_layer2, color="limegreen", linestyle="dashed", linewidth=lineWidth2)

# DenseNet 1 layer
plt.scatter(hs_densenet_growth_rate_vec_layer0, hs_densenet_num_parameters_vec_layer0, color="red",marker="s", label="DenseNet, 1 layer", s=marker_size2)
plt.plot(hs_densenet_growth_rate_vec_layer0, hs_densenet_num_parameters_vec_layer0, color="red", linestyle="dashed", linewidth=lineWidth2)

# DenseNet 2 layers
plt.scatter(hs_densenet_growth_rate_vec_layer1, hs_densenet_num_parameters_vec_layer1, color="darkred",marker="s", label="DenseNet, 2 layers", s=marker_size2)
plt.plot(hs_densenet_growth_rate_vec_layer1, hs_densenet_num_parameters_vec_layer1, color="darkred", linestyle="dashed", linewidth=lineWidth2)

# DenseNet 3 layers
plt.scatter(hs_densenet_growth_rate_vec_layer2, hs_densenet_num_parameters_vec_layer2, color="indianred",marker="s", label="DenseNet, 3 layers", s=marker_size2)
plt.plot(hs_densenet_growth_rate_vec_layer2, hs_densenet_num_parameters_vec_layer2, color="indianred", linestyle="dashed", linewidth=lineWidth2)

# ResNet 1 layer
plt.scatter(hs_resnet_num_neurons_vec_layer0, hs_resnet_num_parameters_vec_layer0,edgecolors='black', color="purple", marker="^", label="ResNet, 1 layer", s=marker_size2)
plt.plot(hs_resnet_num_neurons_vec_layer0, hs_resnet_num_parameters_vec_layer0, color="purple", linestyle="dashed", linewidth=lineWidth2)

# ResNet 2 layers
plt.scatter(hs_resnet_num_neurons_vec_layer1, hs_resnet_num_parameters_vec_layer1,edgecolors='black', color="darkmagenta", marker="^", label="ResNet, 2 layers", s=marker_size2)
plt.plot(hs_resnet_num_neurons_vec_layer1, hs_resnet_num_parameters_vec_layer1, color="darkmagenta", linestyle="dashed", linewidth=lineWidth2)

# ResNet 3 layers
plt.scatter(hs_resnet_num_neurons_vec_layer2, hs_resnet_num_parameters_vec_layer2,edgecolors='black', color="mediumorchid", marker="^", label="ResNet, 3 layers", s=marker_size2)
plt.plot(hs_resnet_num_neurons_vec_layer2, hs_resnet_num_parameters_vec_layer2, color="mediumorchid", linestyle="dashed", linewidth=lineWidth2)

# DNN 1 layer
plt.scatter(hs_dnn_num_neurons_vec_layer0, hs_dnn_num_parameters_vec_layer0, marker="D", color="blue", label="DNN, 1 layer", edgecolors='black', s=marker_size2)
plt.plot(hs_dnn_num_neurons_vec_layer0, hs_dnn_num_parameters_vec_layer0, color="blue", linestyle="dashed", linewidth=lineWidth2)

# DNN 2 layers
plt.scatter(hs_dnn_num_neurons_vec_layer1, hs_dnn_num_parameters_vec_layer1, marker="D", color="darkblue", label="DNN, 2 layers", edgecolors='black', s=marker_size2)
plt.plot(hs_dnn_num_neurons_vec_layer1, hs_dnn_num_parameters_vec_layer1, color="darkblue", linestyle="dashed", linewidth=lineWidth2)

# DNN 3 layers
plt.scatter(hs_dnn_num_neurons_vec_layer2, hs_dnn_num_parameters_vec_layer2, marker="D", color="cornflowerblue", label="DNN, 3 layers", edgecolors='black', s=marker_size2)
plt.plot(hs_dnn_num_neurons_vec_layer2, hs_dnn_num_parameters_vec_layer2, color="cornflowerblue", linestyle="dashed", linewidth=lineWidth2)

# Add labels, title and legend
plt.xlabel('Number of Neurons',fontsize=textSize)  # Fixed variable name
plt.ylabel('Number of Parameters',fontsize=textSize)
# plt.title('Heisenberg Spin System',fontsize=title_size)
plt.xticks([15, 60, 105, 150], labels=["15", "60", "105", "150"],fontsize=xTickSize)
plt.yscale("log")  # Parameters often span orders of magnitude
plt.yticks(fontsize=yTickSize)
plt.legend(loc="best", fontsize=legend_fontsize-10,  # Smaller font
          ncol=2,  # 2 columns instead of 3
          handlelength=0.8,      # Shorter marker lines
          handletextpad=0.2,     # Less space between marker and text
          columnspacing=0.3,     # Tighter column spacing
          borderpad=0.1,         # Less padding
          labelspacing=0.15,     # Less vertical space
          markerscale=0.6)       # Smaller markers
plt.tick_params(axis='both', length=tick_length, width=tick_width)
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width)
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# Improve layout and save
plt.subplots_adjust(left=0.3, right=0.95, top=0.99, bottom=0.15)

# Save the parameter plot
plt.savefig('hs_parameters_vs_neurons.png', dpi=300)
plt.savefig('hs_parameters_vs_neurons.svg')
plt.close()