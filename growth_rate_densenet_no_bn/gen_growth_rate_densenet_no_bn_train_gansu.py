from pathlib import Path
# from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os

#this script generates slurm scripts on gansu
#for training
#densenet, no bn

#growth rate

outPath="./bashFiles_growth_rate_densenet_no_bn/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)

L=15
r=3
chunk_size = 100
num_layers_vec=[2,3,4]
num_neurons_final_est_vec=[30,45,60,75,90,105,120,135,150]

layer_neuron_pairs_vec=[[layer,neuron] for layer in num_layers_vec for neuron in num_neurons_final_est_vec]

chunks=[layer_neuron_pairs_vec[i:i+chunk_size] for i in range(0,len(layer_neuron_pairs_vec),chunk_size)]

def contents_to_bash(num_layers,num_neurons_final_est,file_index):
    growth_rate=int(np.ceil(num_neurons_final_est-L)/(num_layers-1))
    print(f"num_layers={num_layers}, growth_rate={growth_rate}, final_neuron_num={L+(num_layers-1)*growth_rate}")
    contents = [
        "#!/bin/bash\n",
        "#SBATCH -n 5\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-60:00\n",
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=20GB\n",
        f"#SBATCH -o out_growth_rate{growth_rate}_densenet_no_bn_layer{num_layers}.out\n",
        f"#SBATCH -e out_growth_rate{growth_rate}_densenet_no_bn_laye{num_layers}.err\n"
        "cd /public/home/hkust_jwliu_1/liuxi/Documents/pyCode/deep_field_collection/growth_rate_densenet_no_bn\n",
        f"python3 -u growth_rate_denset_no_bn_train.py {L} {r} {num_layers}  {growth_rate}\n",
    ]
    out_chunk = outPath + f"/chunk{file_index // chunk_size}/"
    Path(out_chunk).mkdir(exist_ok=True, parents=True)
    outBashName = out_chunk + f"/train_growth_rate{growth_rate}_densenet_no_bn_layer{num_layers}_neuron{num_neurons}.sh"
    with open(outBashName, "w+") as fptr:
        fptr.writelines(contents)

# Process each pair with its index
for file_index, layer_neuron in enumerate(layer_neuron_pairs_vec):
    num_layers, num_neurons=layer_neuron
    contents_to_bash(num_layers, num_neurons, file_index)