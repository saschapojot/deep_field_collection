from pathlib import Path
# from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os



#this script generates slurm scripts on raccoon
#for training
#quantum Hamiltonian, pbc, vit

outPath="./bashFiles_qt_vit_pbc/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)
num_epochs=1000

layer_vec=[2,3,4,5]
D_vec=[60,70,80,90]

layer_D_vec=[[layer,D] for layer in layer_vec for D in D_vec]
chunk_size = 100

chunks=[layer_D_vec[i:i+chunk_size] for i in range(0,len(layer_D_vec),chunk_size)]


def contents_to_bash(layer,D,file_index):
    contents = [
        "#!/bin/bash\n",
        "#SBATCH -n 1\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-240:00\n",
        "#SBATCH -p CLUSTER\n",
        "#SBATCH --mem=40GB\n",
        f"#SBATCH -o out_train_qt_vit_pbc_layer{layer}_D{D}.out\n",
        f"#SBATCH -e out_train_qt_vit_pbc_layer{layer}_D{D}.err\n",
        "cd /home/cywanag/data/hpc/cywanag/liuxi/Document/pyCode/deep_field_collection/qt_vit\n",
        f"python3 -u qt_vit_train.py {num_epochs} 3 {layer} {D}\n"
        ]
    out_chunk = outPath + f"/chunk{file_index // chunk_size}/"
    Path(out_chunk).mkdir(exist_ok=True, parents=True)
    outBashName = out_chunk + f"/train_qt_vit_pbc_layer{layer}_D{D}.sh"
    with open(outBashName, "w+") as fptr:
        fptr.writelines(contents)


for file_index,layer_D in enumerate(layer_D_vec):
    layer,D=layer_D
    contents_to_bash(layer,D,file_index)