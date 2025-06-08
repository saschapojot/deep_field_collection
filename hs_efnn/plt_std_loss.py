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
epoch_num=15999
layer_num_vec=[1,2,3]

inCsvName_layer0=inPath+f"/layer{layer_num_vec[0]}_epoch{epoch_num}_std_loss.csv"
inCsvName_layer1=inPath+f"/layer{layer_num_vec[1]}_epoch{epoch_num}_std_loss.csv"
inCsvName_layer2=inPath+f"/layer{layer_num_vec[2]}_epoch{epoch_num}_std_loss.csv"


in_df_layer0=pd.read_csv(inCsvName_layer0)
in_df_layer1=pd.read_csv(inCsvName_layer1)
in_df_layer2=pd.read_csv(inCsvName_layer2)

#data layer_num_vec[0]
num_neurons_vec_layer0=np.array(in_df_layer0["num_neurons"])
std_loss_vec_layer0=np.array(in_df_layer0["std_loss"])
#data layer_num_vec[1]
num_neurons_vec_layer1=np.array(in_df_layer1["num_neurons"])
std_loss_vec_layer1=np.array(in_df_layer1["std_loss"])
#data layer_num_vec[2]
num_neurons_vec_layer2=np.array(in_df_layer2["num_neurons"])
std_loss_vec_layer2=np.array(in_df_layer2["std_loss"])
plt.figure()
#layer 0
plt.scatter(num_neurons_vec_layer0,std_loss_vec_layer0,color="green",marker="s",label="1 layers")
plt.plot(num_neurons_vec_layer0,std_loss_vec_layer0,color="green",linestyle="dashed")

#layer 1
plt.scatter(num_neurons_vec_layer1,std_loss_vec_layer1,color="magenta",marker="o",label="2 layers")
plt.plot(num_neurons_vec_layer1,std_loss_vec_layer1,color="magenta",linestyle="dotted")

#layer 2
plt.scatter(num_neurons_vec_layer2,std_loss_vec_layer2,color="navy",marker="P",label="3 layers")
plt.plot(num_neurons_vec_layer2,std_loss_vec_layer2,color="navy",linestyle="dashdot")

plt.xlabel("final Neuron number")
plt.ylabel("Absolute error",fontsize=14)
plt.yscale("log")


plt.title("densenet 2-4 layers, more neurons")
plt.gca().yaxis.set_label_position("right")  # Move label to the right
plt.legend(loc="best")
plt.savefig(inPath+f"/neuron_compare_epoch{epoch_num}.png")
plt.close()