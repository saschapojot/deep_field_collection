
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from itertools import combinations
import matplotlib as mpl
#this script plots test error for
#dnn, efnn, resnet, densenet
#also plots parameter number

L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)

# Set the color explicitly to black
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 2.5  # You already have this

#load result from efnn

efnn_in_path="/home/adada/Documents/pyCode/deep_field/final_sum_more_neurons_inf_range_general_r/compare_layer_neuron_num/"
inDataPath=f"/home/adada/Documents/pyCode/deep_field_collection/densenet_no_bn/data_inf_range_model_L{L}_K_{K}_r{r}/"
in_train_pkl_file=inDataPath+"/inf_range.train.pkl"
with open(in_train_pkl_file, 'rb') as fptr:
    X_train, Y_train = pickle.load(fptr)


abs_avg_Y_train=np.abs(np.mean(np.array(Y_train)))

efnn_layer_num_vec=[1,2,3]
efnn_inCsvName_layer1=efnn_in_path+f"/{efnn_layer_num_vec[0]}_std_loss.csv"
efnn_inCsvName_layer2=efnn_in_path+f"/{efnn_layer_num_vec[1]}_std_loss.csv"
efnn_inCsvName_layer3=efnn_in_path+f"/{efnn_layer_num_vec[2]}_std_loss.csv"

efnn_in_df_layer1=pd.read_csv(efnn_inCsvName_layer1)
efnn_in_df_layer2=pd.read_csv(efnn_inCsvName_layer2)
efnn_in_df_layer3=pd.read_csv(efnn_inCsvName_layer3)
# print(efnn_in_df_layer3)

#efnn data layer 1
efnn_neuron_num_vec_layer1=np.array(efnn_in_df_layer1["neuron_num"])
efnn_std_loss_vec_layer1=np.array(efnn_in_df_layer1["std_loss"])
efnn_num_params_vec_layer1=np.array(efnn_in_df_layer1["num_params"])
efnn_relative_error_layer1=efnn_std_loss_vec_layer1/abs_avg_Y_train
# print(f"efnn_num_params_vec_layer1={efnn_num_params_vec_layer1}")
#efnn data layer 2
efnn_neuron_num_vec_layer2=np.array(efnn_in_df_layer2["neuron_num"])
efnn_std_loss_vec_layer2=np.array(efnn_in_df_layer2["std_loss"])
efnn_num_params_vec_layer2=np.array(efnn_in_df_layer2["num_params"])
efnn_relative_error_layer2=efnn_std_loss_vec_layer2/abs_avg_Y_train

#efnn data layer 3
efnn_neuron_num_vec_layer3=np.array(efnn_in_df_layer3["neuron_num"])
efnn_std_loss_vec_layer3=np.array(efnn_in_df_layer3["std_loss"])
efnn_num_params_vec_layer3=np.array(efnn_in_df_layer3["num_params"])
efnn_relative_error_layer3=efnn_std_loss_vec_layer3/abs_avg_Y_train

#load result from densenet
# densenet_inPath="/home/adada/Documents/pyCode/deep_field_collection/densenet_no_bn/compare_layer_neuron_num/"
densenet_skip_num_vec=[1,2,3]
densenet_inPath="/home/adada/Documents/pyCode/deep_field_collection/final_sum_densenet/compare_layer_neuron_num/"
densenet_skip_num_vec=np.array(densenet_skip_num_vec)

densenet_layer_num_vec=densenet_skip_num_vec
densenet_inCsvName_layer1=densenet_inPath+f"/layer{densenet_layer_num_vec[0]}_std_loss.csv"
densenet_inCsvName_layer2=densenet_inPath+f"/layer{densenet_layer_num_vec[1]}_std_loss.csv"
densenet_inCsvName_layer3=densenet_inPath+f"/layer{densenet_layer_num_vec[2]}_std_loss.csv"

densenet_in_df_layer1=pd.read_csv(densenet_inCsvName_layer1)
densenet_in_df_layer2=pd.read_csv(densenet_inCsvName_layer2)
densenet_in_df_layer3=pd.read_csv(densenet_inCsvName_layer3)

#densenet data layer 1
densenet_neuron_num_vec_layer1=np.array(densenet_in_df_layer1["neuron_num"])
densenet_std_loss_vec_layer1=np.array(densenet_in_df_layer1["std_loss"])
densenet_num_params_vec_layer1=np.array(densenet_in_df_layer1["num_params"])
densenet_relative_error_layer1=densenet_std_loss_vec_layer1/abs_avg_Y_train

#densenet data layer 2
densenet_neuron_num_vec_layer2=np.array(densenet_in_df_layer2["neuron_num"])
densenet_std_loss_vec_layer2=np.array(densenet_in_df_layer2["std_loss"])
densenet_num_params_vec_layer2=np.array(densenet_in_df_layer2["num_params"])
densenet_relative_error_layer2=densenet_std_loss_vec_layer2/abs_avg_Y_train

#densenet data layer 3
densenet_neuron_num_vec_layer3=np.array(densenet_in_df_layer3["neuron_num"])
densenet_std_loss_vec_layer3=np.array(densenet_in_df_layer3["std_loss"])
densenet_num_params_vec_layer3=np.array(densenet_in_df_layer3["num_params"])
densenet_relative_error_layer3=densenet_std_loss_vec_layer3/abs_avg_Y_train

#load result from resnet
# resnet_inPath="/home/adada/Documents/pyCode/deep_field_collection/resnet_no_bn/compare_layer_neuron_num/"
resnet_inPath="/home/adada/Documents/pyCode/deep_field_collection/final_sum_resnet/compare_layer_neuron_num/"
resnet_layer_num_vec=[1,2,3]

resnet_inCsvName_layer1=resnet_inPath+f"/layer{resnet_layer_num_vec[0]}_std_loss.csv"
resnet_inCsvName_layer2=resnet_inPath+f"/layer{resnet_layer_num_vec[1]}_std_loss.csv"
resnet_inCsvName_layer3=resnet_inPath+f"/layer{resnet_layer_num_vec[2]}_std_loss.csv"

resnet_in_df_layer1=pd.read_csv(resnet_inCsvName_layer1)
resnet_in_df_layer2=pd.read_csv(resnet_inCsvName_layer2)
resnet_in_df_layer3=pd.read_csv(resnet_inCsvName_layer3)

#resnet data layer 1
resnet_neuron_num_vec_layer1=np.array(resnet_in_df_layer1["neuron_num"])
resnet_std_loss_vec_layer1=np.array(resnet_in_df_layer1["std_loss"])
resnet_num_params_vec_layer1=np.array(resnet_in_df_layer1["num_params"])
resnet_relative_error_layer1=resnet_std_loss_vec_layer1/abs_avg_Y_train

#resnet data layer 2
resnet_neuron_num_vec_layer2=np.array(resnet_in_df_layer2["neuron_num"])
resnet_std_loss_vec_layer2=np.array(resnet_in_df_layer2["std_loss"])
resnet_num_params_vec_layer2=np.array(resnet_in_df_layer2["num_params"])
resnet_relative_error_layer2=resnet_std_loss_vec_layer2/abs_avg_Y_train


#resnet data layer 3
resnet_neuron_num_vec_layer3=np.array(resnet_in_df_layer3["neuron_num"])
resnet_std_loss_vec_layer3=np.array(resnet_in_df_layer3["std_loss"])
resnet_num_params_vec_layer3=np.array(resnet_in_df_layer3["num_params"])
resnet_relative_error_layer3=resnet_std_loss_vec_layer3/abs_avg_Y_train
# print(f"resnet_num_params_vec_layer3={resnet_num_params_vec_layer3}")


#load result from dnn
dnn_inPath=f"/home/adada/Documents/pyCode/deep_field_collection/final_sum_dnn/compare_layer_neuron_num/"
dnn_epoch_num=15999
dnn_layer_num_vec=[1,2,3]

dnn_inCsvName_layer0=dnn_inPath+f"/layer{dnn_layer_num_vec[0]}_epoch{dnn_epoch_num}_std_loss.csv"
dnn_inCsvName_layer1=dnn_inPath+f"/layer{dnn_layer_num_vec[1]}_epoch{dnn_epoch_num}_std_loss.csv"
dnn_inCsvName_layer2=dnn_inPath+f"/layer{dnn_layer_num_vec[2]}_epoch{dnn_epoch_num}_std_loss.csv"

dnn_in_df_layer0=pd.read_csv(dnn_inCsvName_layer0)
dnn_in_df_layer1=pd.read_csv(dnn_inCsvName_layer1)
dnn_in_df_layer2=pd.read_csv(dnn_inCsvName_layer2)

#dnn, layer 1
dnn_num_neurons_vec_layer0=np.array(dnn_in_df_layer0["num_neurons"])
dnn_std_loss_vec_layer0=np.array(dnn_in_df_layer0["std_loss"])
dnn_num_parameters_vec_layer0=np.array(dnn_in_df_layer0["num_params"])
dnn_relative_error_layer0=dnn_std_loss_vec_layer0/abs_avg_Y_train
print(f"dnn_std_loss_vec_layer0={dnn_std_loss_vec_layer0}")
#dnn, layer 2
dnn_num_neurons_vec_layer1=np.array(dnn_in_df_layer1["num_neurons"])
dnn_std_loss_vec_layer1=np.array(dnn_in_df_layer1["std_loss"])
dnn_num_parameters_vec_layer1=np.array(dnn_in_df_layer1["num_params"])
dnn_relative_error_layer1=dnn_std_loss_vec_layer1/abs_avg_Y_train
#dnn, layer 3
dnn_num_neurons_vec_layer2=np.array(dnn_in_df_layer2["num_neurons"])
dnn_std_loss_vec_layer2=np.array(dnn_in_df_layer2["std_loss"])
dnn_num_parameters_vec_layer2=np.array(dnn_in_df_layer2["num_params"])
dnn_relative_error_layer2=dnn_std_loss_vec_layer2/abs_avg_Y_train

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
# efnn , layer1
plt.scatter(efnn_neuron_num_vec_layer1,efnn_relative_error_layer1,color="green",label="EFNN, 1 layer", s=marker_size2)
plt.plot(efnn_neuron_num_vec_layer1,efnn_relative_error_layer1,color="green",linestyle="dashed", linewidth=lineWidth2)
# efnn , layer2
plt.scatter(efnn_neuron_num_vec_layer2, efnn_relative_error_layer2, color="darkgreen", marker="o", label="EFNN, 2 layers", s=marker_size2)
plt.plot(efnn_neuron_num_vec_layer2, efnn_relative_error_layer2, color="darkgreen", linestyle="dashed", linewidth=lineWidth2)
# efnn , layer3
plt.scatter(efnn_neuron_num_vec_layer3, efnn_relative_error_layer3, color="limegreen", marker="o", label="EFNN, 3 layers", s=marker_size2)
plt.plot(efnn_neuron_num_vec_layer3, efnn_relative_error_layer3, color="limegreen", linestyle="dashed", linewidth=lineWidth2)



# DenseNet,layer1
plt.scatter(densenet_neuron_num_vec_layer1, densenet_relative_error_layer1, color="red", marker="s", label="DenseNet, 1 layer", s=marker_size2)
plt.plot(densenet_neuron_num_vec_layer1, densenet_relative_error_layer1, color="red", linestyle="dashed", linewidth=lineWidth2)
# DenseNet,layer2
plt.scatter(densenet_neuron_num_vec_layer2, densenet_relative_error_layer2, color="darkred", marker="s", label="DenseNet, 2 layers", s=marker_size2)
plt.plot(densenet_neuron_num_vec_layer2, densenet_relative_error_layer2, color="darkred", linestyle="dashed", linewidth=lineWidth2)
# DenseNet,layer3
plt.scatter(densenet_neuron_num_vec_layer3, densenet_relative_error_layer3, color="indianred", marker="s", label="DenseNet, 3 layers", s=marker_size2)
plt.plot(densenet_neuron_num_vec_layer3, densenet_relative_error_layer3, color="indianred", linestyle="dashed", linewidth=lineWidth2)



# ResNet,layer1
plt.scatter(resnet_neuron_num_vec_layer1, resnet_relative_error_layer1,
            color="purple", marker="^", label="ResNet, 1 layer", s=marker_size2)
plt.plot(resnet_neuron_num_vec_layer1, resnet_relative_error_layer1,
         color="purple", linestyle="dashed", linewidth=lineWidth2)

# ResNet,layer2
plt.scatter(resnet_neuron_num_vec_layer2, resnet_relative_error_layer2,
            color="darkmagenta", marker="^", label="ResNet, 2 layers", s=marker_size2)
plt.plot(resnet_neuron_num_vec_layer2, resnet_relative_error_layer2,
         color="darkmagenta", linestyle="dashed", linewidth=lineWidth2)

# ResNet,layer3
plt.scatter(resnet_neuron_num_vec_layer3, resnet_relative_error_layer3,
            color="mediumorchid", marker="^", label="ResNet, 3 layers", s=marker_size2)
plt.plot(resnet_neuron_num_vec_layer3, resnet_relative_error_layer3,
         color="mediumorchid", linestyle="dashed", linewidth=lineWidth2)

#dnn, layer1
plt.scatter(dnn_num_neurons_vec_layer0,dnn_relative_error_layer0,marker="D", color="blue", label="DNN, 1 layer", s=marker_size2)
plt.plot(dnn_num_neurons_vec_layer0,dnn_relative_error_layer0,color="blue", linestyle="dashed", linewidth=lineWidth2)

#dnn, layer2
plt.scatter(dnn_num_neurons_vec_layer1,dnn_relative_error_layer1,marker="D", color="darkblue", label="DNN, 2 layers", s=marker_size2)
plt.plot(dnn_num_neurons_vec_layer1,dnn_relative_error_layer1,color="darkblue", linestyle="dashed", linewidth=lineWidth2)

#dnn, layer3
plt.scatter(dnn_num_neurons_vec_layer2,dnn_relative_error_layer2,marker="D", color="cornflowerblue", label="DNN, 3 layers", s=marker_size2)
plt.plot(dnn_num_neurons_vec_layer2,dnn_relative_error_layer2,color="cornflowerblue", linestyle="dashed", linewidth=lineWidth2)

# Add labels and title
plt.xlabel("Neuron Number",fontsize=textSize)
plt.ylabel("Relative Error",fontsize=textSize)
# plt.title(f"3-spin infinite range model",fontsize=textSize)
plt.yscale("log")  # If your errors span multiple orders of magnitude

# Format y-axis to show values multiplied by 10^2
from matplotlib.ticker import FuncFormatter
def format_func(value, tick_number):
    return f'{value*1e2:.1f}'

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))

# Add the scale text inside the plot area
plt.text(0.02, 1.05, r'$\times 10^{-2}$', transform=plt.gca().transAxes,
         fontsize=textSize, verticalalignment='top')


plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks([15,60,105,150],labels=["15","60","105","150"],fontsize=xTickSize)
plt.yticks(fontsize=yTickSize)
# Add minor ticks for x-axis at intervals of 15
ax = plt.gca()
ax.set_xticks(np.arange(0, 165, 15), minor=True)  # This creates ticks at 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150
# plt.gca().yaxis.set_label_position("right")  # Move label to the right
# Make legend smaller
plt.legend(loc="best", fontsize=legend_fontsize-8,  # Even smaller font
          handlelength=1.0,      # Shorter lines in legend
          handletextpad=0.3,     # Less space between marker and text
          columnspacing=0.5,     # Less space between columns if using columns
          borderpad=0.2,         # Less padding inside legend box
          labelspacing=0.2,      # Less vertical space between entries
          markerscale=0.7)       # Smaller markers in legend
plt.subplots_adjust(left=0.3, right=0.95, top=0.99, bottom=0.15)
plt.tick_params(axis='both', length=tick_length, width=tick_width)
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.subplots_adjust(left=0.3, right=0.95, top=0.90, bottom=0.15)  # Reduced from 0.93 to 0.90
# plt.tight_layout()

plt.savefig("./efnn_vs_all.png", dpi=300,bbox_inches='tight')
plt.savefig("./efnn_vs_all.svg",bbox_inches='tight')
plt.close()


# Plot for number of parameters
plt.figure(figsize=(width, height))
plt.minorticks_on()

# EFNN parameters
plt.scatter(efnn_neuron_num_vec_layer1, efnn_num_params_vec_layer1, color="green", label="EFNN, 1 layer", s=marker_size2)
plt.plot(efnn_neuron_num_vec_layer1, efnn_num_params_vec_layer1, color="green", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(efnn_neuron_num_vec_layer2, efnn_num_params_vec_layer2, color="darkgreen", marker="o", label="EFNN, 2 layers", s=marker_size2)
plt.plot(efnn_neuron_num_vec_layer2, efnn_num_params_vec_layer2, color="darkgreen", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(efnn_neuron_num_vec_layer3, efnn_num_params_vec_layer3, color="limegreen", marker="o", label="EFNN, 3 layers", s=marker_size2)
plt.plot(efnn_neuron_num_vec_layer3, efnn_num_params_vec_layer3, color="limegreen", linestyle="dashed", linewidth=lineWidth2)

# DenseNet parameters
plt.scatter(densenet_neuron_num_vec_layer1, densenet_num_params_vec_layer1, color="red", marker="s", label="DenseNet, 1 layer", s=marker_size2)
plt.plot(densenet_neuron_num_vec_layer1, densenet_num_params_vec_layer1, color="red", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(densenet_neuron_num_vec_layer2, densenet_num_params_vec_layer2, color="darkred", marker="s", label="DenseNet, 2 layers", s=marker_size2)
plt.plot(densenet_neuron_num_vec_layer2, densenet_num_params_vec_layer2, color="darkred", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(densenet_neuron_num_vec_layer3, densenet_num_params_vec_layer3, color="indianred", marker="s", label="DenseNet, 3 layers", s=marker_size2)
plt.plot(densenet_neuron_num_vec_layer3, densenet_num_params_vec_layer3, color="indianred", linestyle="dashed", linewidth=lineWidth2)

# ResNet parameters
plt.scatter(resnet_neuron_num_vec_layer1, resnet_num_params_vec_layer1, color="purple", marker="^", label="ResNet, 1 layer", s=marker_size2)
plt.plot(resnet_neuron_num_vec_layer1, resnet_num_params_vec_layer1, color="purple", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(resnet_neuron_num_vec_layer2, resnet_num_params_vec_layer2, color="darkmagenta", marker="^", label="ResNet, 2 layers", s=marker_size2)
plt.plot(resnet_neuron_num_vec_layer2, resnet_num_params_vec_layer2, color="darkmagenta", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(resnet_neuron_num_vec_layer3, resnet_num_params_vec_layer3, color="mediumorchid", marker="^", label="ResNet, 3 layers", s=marker_size2)
plt.plot(resnet_neuron_num_vec_layer3, resnet_num_params_vec_layer3, color="mediumorchid", linestyle="dashed", linewidth=lineWidth2)

# DNN parameters
plt.scatter(dnn_num_neurons_vec_layer0, dnn_num_parameters_vec_layer0, marker="D", color="blue", label="DNN, 1 layer", s=marker_size2)
plt.plot(dnn_num_neurons_vec_layer0, dnn_num_parameters_vec_layer0, color="blue", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(dnn_num_neurons_vec_layer1, dnn_num_parameters_vec_layer1, marker="D", color="darkblue", label="DNN, 2 layers", s=marker_size2)
plt.plot(dnn_num_neurons_vec_layer1, dnn_num_parameters_vec_layer1, color="darkblue", linestyle="dashed", linewidth=lineWidth2)

plt.scatter(dnn_num_neurons_vec_layer2, dnn_num_parameters_vec_layer2, marker="D", color="cornflowerblue", label="DNN, 3 layers", s=marker_size2)
plt.plot(dnn_num_neurons_vec_layer2, dnn_num_parameters_vec_layer2, color="cornflowerblue", linestyle="dashed", linewidth=lineWidth2)

# Add labels and title
plt.xlabel("Neuron Number", fontsize=textSize)
plt.ylabel("Number of Parameters", fontsize=textSize)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks([15, 60, 105, 150], labels=["15", "60", "105", "150"], fontsize=xTickSize)
plt.yticks(fontsize=yTickSize)
plt.yscale("log")  # If your errors span multiple orders of magnitude

# Make legend smaller
plt.legend(loc="best", fontsize=legend_fontsize-8,
          handlelength=1.0,
          handletextpad=0.3,
          columnspacing=0.5,
          borderpad=0.2,
          labelspacing=0.2,
          markerscale=0.7)

plt.subplots_adjust(left=0.3, right=0.95, top=0.99, bottom=0.15)
plt.tick_params(axis='both', length=tick_length, width=tick_width)
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width)
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.savefig("./params_vs_neurons.png")
plt.savefig("./params_vs_neurons.svg")
plt.close()