from pathlib import Path
# from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os

#this script generates slurm scripts on gansu
#for training
#quantum Hamiltonian, pbc, densenet

outPath="./bashFiles_qt_densenet_pbc/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)
num_epochs=1000
skip_connection1_vec=[1,2,3]

C_vec=[10,15,20,25]
skip_C_vec=[[skip,C] for skip in skip_connection1_vec for C in C_vec]
chunk_size = 100


chunks=[skip_C_vec[i:i+chunk_size] for i in range(0,len(skip_C_vec),chunk_size)]

def contents_to_bash(skip,C,file_index):
    contents = [
        "#!/bin/bash\n",
        "#SBATCH -n 9\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-240:00\n",
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=40GB\n",
        f"#SBATCH -o out_train_qt_densenet_pbc_skip{skip}_C{C}.out\n",
        f"#SBATCH -e out_train_qt_densenet_pbc_skip{skip}_C{C}.err\n",
        "cd /public/home/hkust_jwliu_1/liuxi/Documents/pyCode/deep_field_collection/qt_densenet\n",
        f"python3 -u qt_densenet_train.py {num_epochs} {C} {skip}\n",
    ]
    out_chunk = outPath + f"/chunk{file_index // chunk_size}/"
    Path(out_chunk).mkdir(exist_ok=True, parents=True)
    outBashName = out_chunk + f"/train_qt_densenet_pbc_skip{skip}_C{C}.sh"
    with open(outBashName, "w+") as fptr:
        fptr.writelines(contents)


# Process each pair with its index
for file_index, skip_C in enumerate(skip_C_vec):
    skip,C=skip_C
    contents_to_bash(skip,C,file_index)