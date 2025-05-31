import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from itertools import combinations



#this script plots std_loss/abs(avg)
L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)
inPath="./compare_layer_neuron_num/"
layer_num_vec=[1,2,3]

inCsvName_layer1=inPath+f"/layer{layer_num_vec[0]}_std_loss.csv"
inCsvName_layer2=inPath+f"/layer{layer_num_vec[1]}_std_loss.csv"
inCsvName_layer3=inPath+f"/layer{layer_num_vec[2]}_std_loss.csv"

inDataPath=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"

in_train_pkl_file=inDataPath+"/inf_range.train.pkl"
with open(in_train_pkl_file, 'rb') as fptr:
    X_train, Y_train = pickle.load(fptr)

abs_avg_Y_train=np.abs(np.mean(np.array(Y_train)))
in_df_layer1=pd.read_csv(inCsvName_layer1)
in_df_layer2=pd.read_csv(inCsvName_layer2)
in_df_layer3=pd.read_csv(inCsvName_layer3)

#data layer 1
neuron_num_vec_layer1=np.array(in_df_layer1["neuron_num"])
std_loss_vec_layer1=np.array(in_df_layer1["std_loss"])
relative_error_layer1=std_loss_vec_layer1/abs_avg_Y_train

#data layer 2
neuron_num_vec_layer2=np.array(in_df_layer2["neuron_num"])
std_loss_vec_layer2=np.array(in_df_layer2["std_loss"])
relative_error_layer2=std_loss_vec_layer2/abs_avg_Y_train

#data layer 3
neuron_num_vec_layer3=np.array(in_df_layer3["neuron_num"])
std_loss_vec_layer3=np.array(in_df_layer3["std_loss"])
relative_error_layer3=std_loss_vec_layer3/abs_avg_Y_train

plt.figure()

#layer 1
plt.scatter(neuron_num_vec_layer1,relative_error_layer1,color="green",marker="s",label="1 layer")
plt.plot(neuron_num_vec_layer1,relative_error_layer1,color="green",linestyle="dashed")


#layer 2
plt.scatter(neuron_num_vec_layer2,relative_error_layer2,color="magenta",marker="o",label="2 layers")
plt.plot(neuron_num_vec_layer2,relative_error_layer2,color="magenta",linestyle="dotted")


#layer 3
plt.scatter(neuron_num_vec_layer3,relative_error_layer3,color="navy",marker="P",label="3 layers")
plt.plot(neuron_num_vec_layer3,relative_error_layer3,color="navy",linestyle="dashdot")


plt.xlabel("Neuron number")
plt.ylabel("Relative error",fontsize=14)
plt.title("resnet 1-3 layers, more neurons")
plt.gca().yaxis.set_label_position("right")  # Move label to the right
plt.legend(loc="best")
plt.savefig(inPath+"/neuron_compare.png")
plt.close()