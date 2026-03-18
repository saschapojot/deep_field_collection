
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


# =============================================================================
# Generate CSV tables for relative errors per layer
# =============================================================================

# Layer 1
df_layer1 = pd.DataFrame({
    "EFNN_neuron_num": pd.Series(efnn_neuron_num_vec_layer1),
    "EFNN_relative_error": pd.Series(efnn_relative_error_layer1),
    "DenseNet_neuron_num": pd.Series(densenet_neuron_num_vec_layer1),
    "DenseNet_relative_error": pd.Series(densenet_relative_error_layer1),
    "ResNet_neuron_num": pd.Series(resnet_neuron_num_vec_layer1),
    "ResNet_relative_error": pd.Series(resnet_relative_error_layer1),
    "DNN_neuron_num": pd.Series(dnn_num_neurons_vec_layer0), # Note: DNN uses layer0 for 1 layer
    "DNN_relative_error": pd.Series(dnn_relative_error_layer0)
})
df_layer1.to_csv("./efnn_relative_error_layer1.csv", index=False)
print("Saved relative_error_layer1.csv")

# Layer 2
df_layer2 = pd.DataFrame({
    "EFNN_neuron_num": pd.Series(efnn_neuron_num_vec_layer2),
    "EFNN_relative_error": pd.Series(efnn_relative_error_layer2),
    "DenseNet_neuron_num": pd.Series(densenet_neuron_num_vec_layer2),
    "DenseNet_relative_error": pd.Series(densenet_relative_error_layer2),
    "ResNet_neuron_num": pd.Series(resnet_neuron_num_vec_layer2),
    "ResNet_relative_error": pd.Series(resnet_relative_error_layer2),
    "DNN_neuron_num": pd.Series(dnn_num_neurons_vec_layer1), # Note: DNN uses layer1 for 2 layers
    "DNN_relative_error": pd.Series(dnn_relative_error_layer1)
})
df_layer2.to_csv("./efnn_relative_error_layer2.csv", index=False)
print("Saved relative_error_layer2.csv")

# Layer 3
df_layer3 = pd.DataFrame({
    "EFNN_neuron_num": pd.Series(efnn_neuron_num_vec_layer3),
    "EFNN_relative_error": pd.Series(efnn_relative_error_layer3),
    "DenseNet_neuron_num": pd.Series(densenet_neuron_num_vec_layer3),
    "DenseNet_relative_error": pd.Series(densenet_relative_error_layer3),
    "ResNet_neuron_num": pd.Series(resnet_neuron_num_vec_layer3),
    "ResNet_relative_error": pd.Series(resnet_relative_error_layer3),
    "DNN_neuron_num": pd.Series(dnn_num_neurons_vec_layer2), # Note: DNN uses layer2 for 3 layers
    "DNN_relative_error": pd.Series(dnn_relative_error_layer2)
})
df_layer3.to_csv("./efnn_relative_error_layer3.csv", index=False)
print("Saved relative_error_layer3.csv")
