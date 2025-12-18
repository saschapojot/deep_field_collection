import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from itertools import combinations
#this script plots test error for
#hs_efnn, hs_attn, hs_densenet , hs_longer_attn, hs_resnet

L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)


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

#load result from hs_attn
hs_attn_inPath="/home/adada/Documents/pyCode/deep_field_collection/hs_attn/compare_layer_neuron_num/"
hs_attn_epoch_num=15999
hs_attn_layer_num_vec=[1,2,3]

hs_attn_inCsvName_layer0=hs_attn_inPath+f"/layer{hs_attn_layer_num_vec[0]}_epoch{hs_attn_epoch_num}_std_loss.csv"
hs_attn_inCsvName_layer1=hs_attn_inPath+f"/layer{hs_attn_layer_num_vec[1]}_epoch{hs_attn_epoch_num}_std_loss.csv"
hs_attn_inCsvName_layer2=hs_attn_inPath+f"/layer{hs_attn_layer_num_vec[2]}_epoch{hs_attn_epoch_num}_std_loss.csv"

hs_attn_in_df_layer0=pd.read_csv(hs_attn_inCsvName_layer0)
hs_attn_in_df_layer1=pd.read_csv(hs_attn_inCsvName_layer1)
hs_attn_in_df_layer2=pd.read_csv(hs_attn_inCsvName_layer2)

#hs_attn , layer 1
hs_attn_num_neurons_vec_layer0=np.array(hs_attn_in_df_layer0["num_neurons"])
hs_attn_std_loss_vec_layer0=np.array(hs_attn_in_df_layer0["std_loss"])
hs_attn_num_parameters_vec_layer0=np.array(hs_attn_in_df_layer0["num_params"])
#hs_attn , layer 2
hs_attn_num_neurons_vec_layer1=np.array(hs_attn_in_df_layer1["num_neurons"])
hs_attn_std_loss_vec_layer1=np.array(hs_attn_in_df_layer1["std_loss"])
hs_attn_num_parameters_vec_layer1=np.array(hs_attn_in_df_layer1["num_params"])
#hs_attn , layer 3
hs_attn_num_neurons_vec_layer2=np.array(hs_attn_in_df_layer2["num_neurons"])
hs_attn_std_loss_vec_layer2=np.array(hs_attn_in_df_layer2["std_loss"])
hs_attn_num_parameters_vec_layer2=np.array(hs_attn_in_df_layer2["num_params"])

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
#######
#efnn
plt.figure()
# hs_efnn , 1 layer
plt.scatter(hs_efnn_num_neurons_vec_layer0,hs_efnn_std_loss_vec_layer0,color="green",label="EFNN, 1 layer")
plt.plot(hs_efnn_num_neurons_vec_layer0,hs_efnn_std_loss_vec_layer0,color="green",linestyle="dashed")

# hs_efnn , 2 layers
plt.scatter(hs_efnn_num_neurons_vec_layer1, hs_efnn_std_loss_vec_layer1, color="darkgreen", label="EFNN, 2 layers")
plt.plot(hs_efnn_num_neurons_vec_layer1, hs_efnn_std_loss_vec_layer1, color="darkgreen", linestyle="dashed")

# hs_efnn , 3 layer2
plt.scatter(hs_efnn_num_neurons_vec_layer2, hs_efnn_std_loss_vec_layer2, color="limegreen", label="EFNN, 3 layers")
plt.plot(hs_efnn_num_neurons_vec_layer2, hs_efnn_std_loss_vec_layer2, color="limegreen", linestyle="dashed")


## attention
# hs_attn plots
# plt.scatter(hs_attn_num_neurons_vec_layer0, hs_attn_std_loss_vec_layer0, color="blue", label="Attention, 1 layer")
# plt.plot(hs_attn_num_neurons_vec_layer0, hs_attn_std_loss_vec_layer0, color="blue", linestyle="dashed")

# plt.scatter(hs_attn_num_neurons_vec_layer1, hs_attn_std_loss_vec_layer1, color="darkblue", label="Attention, 2 layers")
# plt.plot(hs_attn_num_neurons_vec_layer1, hs_attn_std_loss_vec_layer1, color="darkblue", linestyle="dashed")

# plt.scatter(hs_attn_num_neurons_vec_layer2, hs_attn_std_loss_vec_layer2, color="royalblue", label="Attention, 3 layers")
# plt.plot(hs_attn_num_neurons_vec_layer2, hs_attn_std_loss_vec_layer2, color="royalblue", linestyle="dashed")

## densenet
# hs_densenet plots
plt.scatter(hs_densenet_growth_rate_vec_layer0, hs_densenet_std_loss_vec_layer0,marker="s", color="red", label="DenseNet, 1 layer")
plt.plot(hs_densenet_growth_rate_vec_layer0, hs_densenet_std_loss_vec_layer0, color="red", linestyle="dashed")

plt.scatter(hs_densenet_growth_rate_vec_layer1, hs_densenet_std_loss_vec_layer1,marker="s", color="darkred", label="DenseNet, 2 layers")
plt.plot(hs_densenet_growth_rate_vec_layer1, hs_densenet_std_loss_vec_layer1, color="darkred", linestyle="dashed")

plt.scatter(hs_densenet_growth_rate_vec_layer2, hs_densenet_std_loss_vec_layer2,marker="s", color="indianred", label="DenseNet, 3 layers")
plt.plot(hs_densenet_growth_rate_vec_layer2, hs_densenet_std_loss_vec_layer2, color="indianred", linestyle="dashed")


#resnet

# hs_resnet plots
plt.scatter(hs_resnet_num_neurons_vec_layer0, hs_resnet_std_loss_vec_layer0,marker="^", color="purple", label="ResNet, 1 layer",edgecolors='black')
plt.plot(hs_resnet_num_neurons_vec_layer0, hs_resnet_std_loss_vec_layer0, color="purple", linestyle="dashed")

plt.scatter(hs_resnet_num_neurons_vec_layer1, hs_resnet_std_loss_vec_layer1,marker="^", color="darkmagenta", label="ResNet, 2 layers",edgecolors='black')
plt.plot(hs_resnet_num_neurons_vec_layer1, hs_resnet_std_loss_vec_layer1, color="darkmagenta", linestyle="dashed")

plt.scatter(hs_resnet_num_neurons_vec_layer2, hs_resnet_std_loss_vec_layer2,marker="^", color="mediumorchid", label="ResNet, 3 layers",edgecolors='black')
plt.plot(hs_resnet_num_neurons_vec_layer2, hs_resnet_std_loss_vec_layer2, color="mediumorchid", linestyle="dashed")
x_font_size=25
y_font_size=25
title_size=20
plt.yscale("log")  # If your errors span multiple orders of magnitude
# Add labels, title and legend
plt.xlabel('Number of Neurons',fontsize=x_font_size)
plt.title(f'Heisenberg Spin System',fontsize=title_size)
plt.xticks([15,60,105,150],labels=["15","60","105","150"],fontsize=x_font_size)
plt.ylabel("Absolute error",fontsize=y_font_size)
plt.legend(loc='best')
plt.yticks(fontsize=y_font_size)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
# Improve layout and save
plt.tight_layout()
plt.savefig('neural_network_comparison.png', dpi=300)
plt.savefig('neural_network_comparison.svg')
plt.close()


# Plot number of parameters
plt.figure()


# EFNN 1 layer
plt.scatter(hs_efnn_num_neurons_vec_layer0, hs_efnn_num_parameters_vec_layer0, color="green",marker="o", label="EFNN, 1 layer")
plt.plot(hs_efnn_num_neurons_vec_layer0, hs_efnn_num_parameters_vec_layer0, color="green", linestyle="dashed")

# EFNN 2 layers
plt.scatter(hs_efnn_num_neurons_vec_layer1, hs_efnn_num_parameters_vec_layer1, color="darkgreen",marker="o", label="EFNN, 2 layers")
plt.plot(hs_efnn_num_neurons_vec_layer1, hs_efnn_num_parameters_vec_layer1, color="darkgreen", linestyle="dashed")

# EFNN 3 layers
plt.scatter(hs_efnn_num_neurons_vec_layer2, hs_efnn_num_parameters_vec_layer2, color="limegreen",marker="o", label="EFNN, 3 layers")
plt.plot(hs_efnn_num_neurons_vec_layer2, hs_efnn_num_parameters_vec_layer2, color="limegreen", linestyle="dashed")

# DenseNet 1 layer
plt.scatter(hs_densenet_growth_rate_vec_layer0, hs_densenet_num_parameters_vec_layer0, color="red",marker="s", label="DenseNet, 1 layer")
plt.plot(hs_densenet_growth_rate_vec_layer0, hs_densenet_num_parameters_vec_layer0, color="red", linestyle="dashed")

# DenseNet 2 layers
plt.scatter(hs_densenet_growth_rate_vec_layer1, hs_densenet_num_parameters_vec_layer1, color="darkred",marker="s", label="DenseNet, 2 layers")
plt.plot(hs_densenet_growth_rate_vec_layer1, hs_densenet_num_parameters_vec_layer1, color="darkred", linestyle="dashed")

# DenseNet 3 layers
plt.scatter(hs_densenet_growth_rate_vec_layer2, hs_densenet_num_parameters_vec_layer2, color="indianred",marker="s", label="DenseNet, 3 layers")
plt.plot(hs_densenet_growth_rate_vec_layer2, hs_densenet_num_parameters_vec_layer2, color="indianred", linestyle="dashed")

# ResNet 1 layer
plt.scatter(hs_resnet_num_neurons_vec_layer0, hs_resnet_num_parameters_vec_layer0,edgecolors='black', color="purple", marker="^", label="ResNet, 1 layer")
plt.plot(hs_resnet_num_neurons_vec_layer0, hs_resnet_num_parameters_vec_layer0, color="purple", linestyle="dashed")

# ResNet 2 layers
plt.scatter(hs_resnet_num_neurons_vec_layer1, hs_resnet_num_parameters_vec_layer1,edgecolors='black', color="darkmagenta", marker="^", label="ResNet, 2 layers")
plt.plot(hs_resnet_num_neurons_vec_layer1, hs_resnet_num_parameters_vec_layer1, color="darkmagenta", linestyle="dashed")

# ResNet 3 layers
plt.scatter(hs_resnet_num_neurons_vec_layer2, hs_resnet_num_parameters_vec_layer2,edgecolors='black', color="mediumorchid", marker="^", label="ResNet, 3 layers")
plt.plot(hs_resnet_num_neurons_vec_layer2, hs_resnet_num_parameters_vec_layer2, color="mediumorchid", linestyle="dashed")


# Add labels, title and legend
plt.xlabel('Number of Neurons',fontsize=x_font_size)
plt.ylabel('Number of Parameters',fontsize=y_font_size)
plt.title('Heisenberg Spin System',fontsize=title_size)
plt.xticks([15, 60, 105, 150], labels=["15", "60", "105", "150"],fontsize=x_font_size)
plt.yscale("log")  # Parameters often span orders of magnitude
plt.yticks(fontsize=y_font_size)
plt.legend(loc='best')
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# Improve layout and save
plt.tight_layout()
plt.savefig('neural_network_parameter_comparison.png', dpi=300)
plt.savefig('neural_network_parameter_comparison.svg')