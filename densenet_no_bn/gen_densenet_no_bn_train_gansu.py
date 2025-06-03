from pathlib import Path
# from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os

#this script generates slurm scripts on gansu
#for training
#densenet, no bn


outPath="./bashFiles_densenet_no_bn/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)


L=15
r=3
chunk_size = 100
num_skip_connections_vec=[3]
num_neurons_vec=[15,30,45,60,75,90,105,120,135,150]

skip_neuron_pairs_vec=[[sk,neuron] for sk in num_skip_connections_vec for neuron in num_neurons_vec ]

chunks=[skip_neuron_pairs_vec[i:i+chunk_size] for i in range(0,len(skip_neuron_pairs_vec),chunk_size)]

def contents_to_bash(num_skip,num_neurons,file_index):
    contents = [
        "#!/bin/bash\n",
        "#SBATCH -n 1\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-120:00\n",
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=40GB\n",
        f"#SBATCH -o out_train_densenet_no_bn_skip{num_skip}_neuron{num_neurons}.out\n",
        f"#SBATCH -e out_train_densenet_no_bn_skip{num_skip}_neuron{num_neurons}.err\n",
        f"cd /public/home/hkust_jwliu_1/liuxi/Documents/pyCode/deep_field_collection/densenet_no_bn\n",
        f"python3 -u denset_no_bn_train.py {L} {r} {num_skip} {num_neurons}\n"
    ]
    out_chunk = outPath + f"/chunk{file_index // chunk_size}/"
    Path(out_chunk).mkdir(exist_ok=True, parents=True)
    outBashName = out_chunk + f"/train_densenet_no_bn_skip{num_skip}_neuron{num_neurons}.sh"
    with open(outBashName, "w+") as fptr:
        fptr.writelines(contents)


# Process each pair with its index
for file_index, skip_neuron in enumerate(skip_neuron_pairs_vec):
    num_skip, num_neurons=skip_neuron
    contents_to_bash(num_skip, num_neurons, file_index)