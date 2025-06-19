from pathlib import Path
# from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os

#this script generates slurm scripts on gansu

outPath="./bashFiles_hs_longer_attn/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)

L=15
r=3
chunk_size = 100
num_layers_vec=[1,2,3]
num_heads=5
embed_dim_vec=[15,30,45,60,75,90,105,120,135,150]

layer_embed_dim_pairs_vec=[[layer,embed] for layer in num_layers_vec for embed in embed_dim_vec]

chunks=[layer_embed_dim_pairs_vec[i:i+chunk_size] for i in range(0,len(layer_embed_dim_pairs_vec),chunk_size)]

def contents_to_bash(layer,embed,file_index):
    contents = [
        "#!/bin/bash\n",
        "#SBATCH -n 5\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-60:00\n",
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=20GB\n",
        f"#SBATCH -o out_longer_attn_layer{layer}_embed{embed}.out\n",
        f"#SBATCH -e out_longer_attn_layer{layer}_embed{embed}.err\n",
        f"cd /public/home/hkust_jwliu_1/liuxi/Documents/pyCode/deep_field_collection/hs_longer_attn\n",
        f"python3 -u hs_longer_attn_train.py {L} {r} {layer} {embed} {num_heads}\n",
    ]
    out_chunk = outPath + f"/chunk{file_index // chunk_size}/"
    Path(out_chunk).mkdir(exist_ok=True, parents=True)
    outBashName = out_chunk + f"/train_hs_longer_attn_layer{layer}_embed{embed}_head{num_heads}.sh"
    with open(outBashName, "w+") as fptr:
        fptr.writelines(contents)

# Process each pair with its index
for file_index, layer_embed in enumerate(layer_embed_dim_pairs_vec):
    layer,embed=layer_embed
    contents_to_bash(layer,embed,file_index)
