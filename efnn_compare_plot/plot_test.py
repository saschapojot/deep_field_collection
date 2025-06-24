
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from itertools import combinations
#this script plots test error for
#efnn, resnet, densenet

L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)
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
efnn_relative_error_layer1=efnn_std_loss_vec_layer1/abs_avg_Y_train

#efnn data layer 2
efnn_neuron_num_vec_layer2=np.array(efnn_in_df_layer2["neuron_num"])
efnn_std_loss_vec_layer2=np.array(efnn_in_df_layer2["std_loss"])
efnn_relative_error_layer2=efnn_std_loss_vec_layer2/abs_avg_Y_train

#efnn data layer 3
efnn_neuron_num_vec_layer3=np.array(efnn_in_df_layer3["neuron_num"])
efnn_std_loss_vec_layer3=np.array(efnn_in_df_layer3["std_loss"])
efnn_relative_error_layer3=efnn_std_loss_vec_layer3/abs_avg_Y_train

#load result from densenet
densenet_inPath="/home/adada/Documents/pyCode/deep_field_collection/densenet_no_bn/compare_layer_neuron_num/"
densenet_skip_num_vec=[1,2,3]
densenet_skip_num_vec=np.array(densenet_skip_num_vec)

densenet_layer_num_vec=densenet_skip_num_vec+1
densenet_inCsvName_layer1=densenet_inPath+f"/layer{densenet_layer_num_vec[0]}_std_loss.csv"
densenet_inCsvName_layer2=densenet_inPath+f"/layer{densenet_layer_num_vec[1]}_std_loss.csv"
densenet_inCsvName_layer3=densenet_inPath+f"/layer{densenet_layer_num_vec[2]}_std_loss.csv"

densenet_in_df_layer1=pd.read_csv(densenet_inCsvName_layer1)
densenet_in_df_layer2=pd.read_csv(densenet_inCsvName_layer2)
densenet_in_df_layer3=pd.read_csv(densenet_inCsvName_layer3)

#densenet data layer 1
densenet_neuron_num_vec_layer1=np.array(densenet_in_df_layer1["neuron_num"])
densenet_std_loss_vec_layer1=np.array(densenet_in_df_layer1["std_loss"])
densenet_relative_error_layer1=densenet_std_loss_vec_layer1/abs_avg_Y_train

#densenet data layer 2
densenet_neuron_num_vec_layer2=np.array(densenet_in_df_layer2["neuron_num"])
densenet_std_loss_vec_layer2=np.array(densenet_in_df_layer2["std_loss"])
densenet_relative_error_layer2=densenet_std_loss_vec_layer2/abs_avg_Y_train

#densenet data layer 3
densenet_neuron_num_vec_layer3=np.array(densenet_in_df_layer3["neuron_num"])
densenet_std_loss_vec_layer3=np.array(densenet_in_df_layer3["std_loss"])
densenet_relative_error_layer3=densenet_std_loss_vec_layer3/abs_avg_Y_train

#load result from resnet
resnet_inPath="/home/adada/Documents/pyCode/deep_field_collection/resnet_no_bn/compare_layer_neuron_num/"
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
resnet_relative_error_layer1=resnet_std_loss_vec_layer1/abs_avg_Y_train

#resnet data layer 2
resnet_neuron_num_vec_layer2=np.array(resnet_in_df_layer2["neuron_num"])
resnet_std_loss_vec_layer2=np.array(resnet_in_df_layer2["std_loss"])
resnet_relative_error_layer2=resnet_std_loss_vec_layer2/abs_avg_Y_train


#resnet data layer 3
resnet_neuron_num_vec_layer3=np.array(resnet_in_df_layer3["neuron_num"])
resnet_std_loss_vec_layer3=np.array(resnet_in_df_layer3["std_loss"])
resnet_relative_error_layer3=resnet_std_loss_vec_layer3/abs_avg_Y_train

#######

plt.figure()
# efnn , layer1
plt.scatter(efnn_neuron_num_vec_layer1,efnn_relative_error_layer1,color="green",label="EFNN, 1 layer")
plt.plot(efnn_neuron_num_vec_layer1,efnn_relative_error_layer1,color="green",linestyle="dashed")
# efnn , layer2
plt.scatter(efnn_neuron_num_vec_layer2, efnn_relative_error_layer2, color="blue", marker="o", label="EFNN, 2 layers")
plt.plot(efnn_neuron_num_vec_layer2, efnn_relative_error_layer2, color="blue", linestyle="dashed")
# efnn , layer3
plt.scatter(efnn_neuron_num_vec_layer3, efnn_relative_error_layer3, color="cyan", marker="o", label="EFNN, 3 layers")
plt.plot(efnn_neuron_num_vec_layer3, efnn_relative_error_layer3, color="cyan", linestyle="dashed")



# DenseNet,layer1
plt.scatter(densenet_neuron_num_vec_layer1, densenet_relative_error_layer1, color="red", marker="s", label="DenseNet, 1 layer")
plt.plot(densenet_neuron_num_vec_layer1, densenet_relative_error_layer1, color="red", linestyle="dashed")
# DenseNet,layer2
plt.scatter(densenet_neuron_num_vec_layer2, densenet_relative_error_layer2, color="magenta", marker="s", label="DenseNet, 2 layers")
plt.plot(densenet_neuron_num_vec_layer2, densenet_relative_error_layer2, color="magenta", linestyle="dashed")
# DenseNet,layer3
plt.scatter(densenet_neuron_num_vec_layer3, densenet_relative_error_layer3, color="pink", marker="s", label="DenseNet, 3 layers")
plt.plot(densenet_neuron_num_vec_layer3, densenet_relative_error_layer3, color="pink", linestyle="dashed")



# ResNet,layer1
plt.scatter(resnet_neuron_num_vec_layer1, resnet_relative_error_layer1,
            color="darkred", marker="^", s=80, label="ResNet, 1 layer",
            edgecolors='black', linewidths=1, alpha=0.8)
plt.plot(resnet_neuron_num_vec_layer1, resnet_relative_error_layer1,
         color="darkred", linestyle="-", linewidth=2, alpha=0.8)

# ResNet,layer2
plt.scatter(resnet_neuron_num_vec_layer2, resnet_relative_error_layer2,
            color="orange", marker="o", s=80, label="ResNet, 2 layers",
            edgecolors='black', linewidths=1, alpha=0.6)
plt.plot(resnet_neuron_num_vec_layer2, resnet_relative_error_layer2,
         color="orange", linestyle="--", linewidth=2, alpha=0.8)

# ResNet,layer3
plt.scatter(resnet_neuron_num_vec_layer3, resnet_relative_error_layer3,
            color="gold", marker="s", s=80, label="ResNet, 3 layers",
            edgecolors='black', linewidths=1, alpha=0.4)
plt.plot(resnet_neuron_num_vec_layer3, resnet_relative_error_layer3,
         color="gold", linestyle=":", linewidth=2, alpha=0.8)

# Add labels and title
plt.xlabel("Number of Neurons")
plt.ylabel("Relative Error")
plt.title(f"Test Error Comparison for 3-spin infinite range model")
plt.yscale("log")  # If your errors span multiple orders of magnitude
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks([15,60,105,150],labels=["15","60","105","150"],)

plt.gca().yaxis.set_label_position("right")  # Move label to the right
plt.legend(loc="best", fontsize="small")
plt.tight_layout()
plt.savefig("./neuron_compare_all.png", dpi=300)
plt.close()