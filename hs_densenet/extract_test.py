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
epoch_num=5999

growth_rate_dirs_vec=[]
growth_rate_vals_vec=[]
dirs_root=f"./out_model_hs_densenet_L{L}_K{K}_r{r}_layer{num_layers}/"

for dir in glob.glob(dirs_root+f"/growth_rate*"):
    match_grow_rate=re.search(r"growth_rate(\d+)", dir)
    if match_grow_rate:
        growth_rate_dirs_vec.append(dir)
        growth_rate_vals_vec.append(int(match_grow_rate.group(1)))


sorted_inds=np.argsort(growth_rate_vals_vec)
sorted_growth_rate_vec=[growth_rate_vals_vec[ind] for ind in sorted_inds]
sorted_growth_rate_dirs_vec=[growth_rate_dirs_vec[ind] for ind in sorted_inds]
pattern_std=r'std_loss\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'

def load_one_test_set_result(dir):
    test_result_file=dir+f"/test_epoch{epoch_num}.txt"
    with open(test_result_file,"r") as fptr:
        line=fptr.readline()
    match_std_loss = re.search(pattern_std, line)
    if match_std_loss:
        return float(match_std_loss.group(1))
    else:
        print(f"{test_result_file}, format error")
        exit(12)

outPath="./compare_layer_neuron_num/"
Path(outPath).mkdir(parents=True, exist_ok=True)
std_loss_vec=[]
for ind in range(0,len(sorted_growth_rate_vec)):
    growth_rate=sorted_growth_rate_vec[ind]
    dir=sorted_growth_rate_dirs_vec[ind]
    std_loss=load_one_test_set_result(dir)
    std_loss_vec.append(std_loss)


out_df=pd.DataFrame({
"growth_rate":sorted_growth_rate_vec,
    "std_loss":std_loss_vec
})
out_df.to_csv(outPath+f"layer{num_layers}_epoch{epoch_num}_std_loss.csv",index=False)