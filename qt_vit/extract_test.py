import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from itertools import combinations
import glob

#this script combines test results for the same num_layers, all D

num_layers=4
epoch_num=700
C=3
D_dirs_vec=[]
D_vals_vec=[]

dirs_root=f"./out_model_data/N10/C{C}/layer{num_layers}/"
for dir in glob.glob(dirs_root+f"/D*"):
    match_D = re.search(r"D(\d+)", dir)
    if match_D:
        D_dirs_vec.append(dir)
        D_vals_vec.append(int(match_D.group(1)))

sorted_inds=np.argsort(D_vals_vec)
sorted_D_vals_vec=[D_vals_vec[ind] for ind in sorted_inds]
sorted_D_dirs_vec=[D_dirs_vec[ind ] for ind in sorted_inds]

epoch_pattern=r"num_epochs=(\d+)"
pattern_std=r'std_loss\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
pattern_num_params=r"num_params\s*=\s*(\d+)"

def match_line_in_file(dir,epoch_num):
    test_fileName = dir + f"/test_over_epochs.txt"
    with open(test_fileName,'r') as fptr:
        contents=fptr.readlines()
        for line in contents:
            match_epoch = re.search(epoch_pattern, line)
            if match_epoch:
                epoch_in_file = int(match_epoch.group(1))
                if epoch_in_file == epoch_num:
                    match_std = re.search(pattern_std, line)
                    match_num_params=re.search(pattern_num_params,line)
                    return epoch_in_file, float(match_std.group(1)), int(match_num_params.group(1))

                else:
                    continue


outPath="./out_model_data/"
Path(outPath).mkdir(parents=True, exist_ok=True)
std_loss_vec=[]
num_params_vec=[]
for ind in range(0,len(sorted_D_vals_vec)):
    D=sorted_D_vals_vec[ind]
    dir =sorted_D_dirs_vec[ind]
    ep,std,num_params=match_line_in_file(dir,epoch_num)
    std_loss_vec.append(std)
    num_params_vec.append(num_params)


out_df=pd.DataFrame({

    "D":sorted_D_vals_vec,
"std_loss":std_loss_vec,
"num_params": num_params_vec
})
out_df.to_csv(outPath+f"layer{num_layers}_epoch{epoch_num}_std_loss.csv",index=False)