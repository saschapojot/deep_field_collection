import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from itertools import combinations
import glob



#this script combines test results for the same num_layers, all num_neurons
L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)
num_layers=3

epoch_num=15999
num_neurons_dirs_vec=[]
num_neurons_vals_vec=[]
dirs_root=f"./out_model_final_sum_resnet_L{L}_K{K}_r{r}_layer{num_layers}/"

for dir in glob.glob(dirs_root+f"/neuron*"):
    match_neuron = re.search(r"neuron(\d+)", dir)
    if match_neuron:
        num_neurons_dirs_vec.append(dir)
        num_neurons_vals_vec.append(int(match_neuron.group(1)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

sorted_inds=np.argsort(num_neurons_vals_vec)
sorted_num_neurons_vec=[num_neurons_vals_vec[ind] for ind in sorted_inds]
sorted_num_neurons_dirs_vec=[num_neurons_dirs_vec[ind] for ind in sorted_inds]

pattern_std=r'std_loss\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
pattern_num_params=r"num_params\s*=\s*(\d+)"
def load_one_test_set_result(dir):
    test_result_file = dir + f"/test_epoch{epoch_num}.txt"
    with open(test_result_file,"r") as fptr:
        line=fptr.readline()
    match_std_loss = re.search(pattern_std, line)
    if match_std_loss:
        return float(match_std_loss.group(1))
    else:
        print(f"{test_result_file}, format error")
        exit(12)

def file_2_num_params(dir):
    test_fileName=dir+f"/test_epoch{epoch_num}.txt"
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()
    match_num_params = re.search(pattern_num_params, line)
    if match_num_params:
        return int(match_num_params.group(1))
    else:
        print(f"{test_fileName}, format error")
        exit(12)
outPath="./compare_layer_neuron_num/"
Path(outPath).mkdir(parents=True, exist_ok=True)
std_loss_vec=[]
num_params_vec=[]
for ind in range(0,len(sorted_num_neurons_vec)):
    num_neurons=sorted_num_neurons_vec[ind]
    dir=sorted_num_neurons_dirs_vec[ind]
    std_loss = load_one_test_set_result(dir)
    std_loss_vec.append(std_loss)
    num_params = file_2_num_params(dir)
    num_params_vec.append(num_params)


out_df=pd.DataFrame({
"num_neurons":sorted_num_neurons_vec,
    "std_loss":std_loss_vec,
"num_params": num_params_vec
})
out_df.to_csv(outPath+f"layer{num_layers}_epoch{epoch_num}_std_loss.csv",index=False)