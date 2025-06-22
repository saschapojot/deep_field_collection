import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from itertools import combinations
import glob

#this script combines test results for the same num_layers, all embed_dim

L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)
num_layers=3
epoch_num=15999

num_embed_dim_vec=[]
num_embed_dim_dirs_vec=[]
dirs_root=f"./out_model_hs_attn_L{L}_K{K}_r3_layer{num_layers}/"

for dir in glob.glob(dirs_root+f"/embed_dim*"):
    match_neuron_num=re.search(r"embed_dim(\d+)", dir)
    if match_neuron_num:
        num_embed_dim_dirs_vec.append(dir)
        num_embed_dim_vec.append(int(match_neuron_num.group(1)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

sorted_inds=np.argsort(num_embed_dim_vec)
sorted_num_embed_dim_vec=[num_embed_dim_vec[ind] for ind in sorted_inds]
sorted_num_embed_dim_dirs_vec=[num_embed_dim_dirs_vec[ind] for ind in sorted_inds]

pattern_std=r'std_loss\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'

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


outPath="./compare_layer_embed_dim/"
Path(outPath).mkdir(parents=True, exist_ok=True)
std_loss_vec=[]
for ind in range(0,len(sorted_num_embed_dim_vec)):
    num_neurons=sorted_num_embed_dim_vec[ind]
    dir=sorted_num_embed_dim_dirs_vec[ind]
    std_loss = load_one_test_set_result(dir)
    std_loss_vec.append(std_loss)

out_df=pd.DataFrame({
"num_neurons":sorted_num_embed_dim_vec,
    "std_loss":std_loss_vec
})
out_df.to_csv(outPath+f"layer{num_layers}_epoch{epoch_num}_std_loss.csv",index=False)