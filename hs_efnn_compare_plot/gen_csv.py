import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from itertools import combinations
import matplotlib as mpl
#this script plots test error for
#hs_efnn, hs_dnn, hs_densenet , hs_resnet

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



#load result from final sum hs_densenet
hs_densenet_inPath="/home/adada/Documents/pyCode/deep_field_collection/final_sum_hs_densenet/compare_layer_neuron_num/"

hs_densenet_epoch_num=15999
hs_densenet_layer_num_vec=[1,2,3]

hs_densenet_inCsvName_layer0=hs_densenet_inPath+f"/layer{hs_densenet_layer_num_vec[0]}_epoch{hs_densenet_epoch_num}_std_loss.csv"
hs_densenet_inCsvName_layer1=hs_densenet_inPath+f"/layer{hs_densenet_layer_num_vec[1]}_epoch{hs_densenet_epoch_num}_std_loss.csv"
hs_densenet_inCsvName_layer2=hs_densenet_inPath+f"/layer{hs_densenet_layer_num_vec[2]}_epoch{hs_densenet_epoch_num}_std_loss.csv"



hs_densenet_in_df_layer0=pd.read_csv(hs_densenet_inCsvName_layer0)
hs_densenet_in_df_layer1=pd.read_csv(hs_densenet_inCsvName_layer1)
hs_densenet_in_df_layer2=pd.read_csv(hs_densenet_inCsvName_layer2)

#hs_densenet , layer 1
hs_densenet_growth_rate_vec_layer0=np.array(hs_densenet_in_df_layer0["neuron_num"])
hs_densenet_std_loss_vec_layer0=np.array(hs_densenet_in_df_layer0["std_loss"])
hs_densenet_num_parameters_vec_layer0=np.array(hs_densenet_in_df_layer0["num_params"])
#hs_densenet , layer 2
hs_densenet_growth_rate_vec_layer1=np.array(hs_densenet_in_df_layer1["neuron_num"])
hs_densenet_std_loss_vec_layer1=np.array(hs_densenet_in_df_layer1["std_loss"])
hs_densenet_num_parameters_vec_layer1=np.array(hs_densenet_in_df_layer1["num_params"])
#hs_densenet , layer 3
hs_densenet_growth_rate_vec_layer2=np.array(hs_densenet_in_df_layer2["neuron_num"])
hs_densenet_std_loss_vec_layer2=np.array(hs_densenet_in_df_layer2["std_loss"])
hs_densenet_num_parameters_vec_layer2=np.array(hs_densenet_in_df_layer2["num_params"])

#load result from final sum hs_resnet
#hs_resnet , layer 1
hs_resnet_inPath="/home/adada/Documents/pyCode/deep_field_collection/final_sum_hs_resnet/compare_layer_neuron_num/"

hs_resnet_epoch_num=15999
hs_resnet_layer_num_vec=[1,2,3]

hs_resnet_inCsvName_layer0=hs_resnet_inPath+f"/layer{hs_resnet_layer_num_vec[0]}_epoch{hs_resnet_epoch_num}_std_loss.csv"
hs_resnet_inCsvName_layer1=hs_resnet_inPath+f"/layer{hs_resnet_layer_num_vec[1]}_epoch{hs_resnet_epoch_num}_std_loss.csv"
hs_resnet_inCsvName_layer2=hs_resnet_inPath+f"/layer{hs_resnet_layer_num_vec[2]}_epoch{hs_resnet_epoch_num}_std_loss.csv"


hs_resnet_in_df_layer0=pd.read_csv(hs_resnet_inCsvName_layer0)
hs_resnet_in_df_layer1=pd.read_csv(hs_resnet_inCsvName_layer1)
hs_resnet_in_df_layer2=pd.read_csv(hs_resnet_inCsvName_layer2)

#hs_resnet , layer 1
hs_resnet_num_neurons_vec_layer0=np.array(hs_resnet_in_df_layer0["neuron_num"])
hs_resnet_std_loss_vec_layer0=np.array(hs_resnet_in_df_layer0["std_loss"])
hs_resnet_num_parameters_vec_layer0=np.array(hs_resnet_in_df_layer0["num_params"])
#hs_resnet , layer 2
hs_resnet_num_neurons_vec_layer1=np.array(hs_resnet_in_df_layer1["neuron_num"])
hs_resnet_std_loss_vec_layer1=np.array(hs_resnet_in_df_layer1["std_loss"])
hs_resnet_num_parameters_vec_layer1=np.array(hs_resnet_in_df_layer1["num_params"])
#hs_resnet , layer 3
hs_resnet_num_neurons_vec_layer2=np.array(hs_resnet_in_df_layer2["neuron_num"])
hs_resnet_std_loss_vec_layer2=np.array(hs_resnet_in_df_layer2["std_loss"])
hs_resnet_num_parameters_vec_layer2=np.array(hs_resnet_in_df_layer2["num_params"])

#load result from hs_dnn

hs_dnn_inPath=f"/home/adada/Documents/pyCode/deep_field_collection/final_sum_hs_dnn/compare_layer_neuron_num/"
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


# =============================================================================
# Generate CSV tables for RMSE loss per layer
# =============================================================================

# Layer 1 (using layer0 variables)
df_layer1 = pd.DataFrame({
    "EFNN_neuron_num": pd.Series(hs_efnn_num_neurons_vec_layer0),
    "EFNN_RMSE_loss": pd.Series(hs_efnn_std_loss_vec_layer0),
    "DenseNet_neuron_num": pd.Series(hs_densenet_growth_rate_vec_layer0),
    "DenseNet_RMSE_loss": pd.Series(hs_densenet_std_loss_vec_layer0),
    "ResNet_neuron_num": pd.Series(hs_resnet_num_neurons_vec_layer0),
    "ResNet_RMSE_loss": pd.Series(hs_resnet_std_loss_vec_layer0),
    "DNN_neuron_num": pd.Series(hs_dnn_num_neurons_vec_layer0),
    "DNN_RMSE_loss": pd.Series(hs_dnn_std_loss_vec_layer0)
})
df_layer1.to_csv("./hs_RMSE_loss_layer1.csv", index=False)
print("Saved hs_RMSE_loss_layer1.csv")

# Layer 2 (using layer1 variables)
df_layer2 = pd.DataFrame({
    "EFNN_neuron_num": pd.Series(hs_efnn_num_neurons_vec_layer1),
    "EFNN_RMSE_loss": pd.Series(hs_efnn_std_loss_vec_layer1),
    "DenseNet_neuron_num": pd.Series(hs_densenet_growth_rate_vec_layer1),
    "DenseNet_RMSE_loss": pd.Series(hs_densenet_std_loss_vec_layer1),
    "ResNet_neuron_num": pd.Series(hs_resnet_num_neurons_vec_layer1),
    "ResNet_RMSE_loss": pd.Series(hs_resnet_std_loss_vec_layer1),
    "DNN_neuron_num": pd.Series(hs_dnn_num_neurons_vec_layer1),
    "DNN_RMSE_loss": pd.Series(hs_dnn_std_loss_vec_layer1)
})
df_layer2.to_csv("./hs_RMSE_loss_layer2.csv", index=False)
print("Saved hs_RMSE_loss_layer2.csv")

# Layer 3 (using layer2 variables)
df_layer3 = pd.DataFrame({
    "EFNN_neuron_num": pd.Series(hs_efnn_num_neurons_vec_layer2),
    "EFNN_RMSE_loss": pd.Series(hs_efnn_std_loss_vec_layer2),
    "DenseNet_neuron_num": pd.Series(hs_densenet_growth_rate_vec_layer2),
    "DenseNet_RMSE_loss": pd.Series(hs_densenet_std_loss_vec_layer2),
    "ResNet_neuron_num": pd.Series(hs_resnet_num_neurons_vec_layer2),
    "ResNet_RMSE_loss": pd.Series(hs_resnet_std_loss_vec_layer2),
    "DNN_neuron_num": pd.Series(hs_dnn_num_neurons_vec_layer2),
    "DNN_RMSE_loss": pd.Series(hs_dnn_std_loss_vec_layer2)
})
df_layer3.to_csv("./hs_RMSE_loss_layer3.csv", index=False)
print("Saved hs_RMSE_loss_layer3.csv")