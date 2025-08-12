from pathlib import Path
import shutil
import os
#this script generates slurm scripts on gansu
#for training
#quantum Hamiltonian, pbc

outPath="./bashFiles_fs_qt_resnet_pbc/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)
num_epochs=1000
step_num_after_S1_vec=[0,1,2]

C_vec=[1,3,5,7,9]

layer_C_pairs_vec=[[layer,C] for layer in step_num_after_S1_vec for C in C_vec]
chunk_size = 100
chunks=[layer_C_pairs_vec[i:i+chunk_size] for i in range(0,len(layer_C_pairs_vec),chunk_size)]
def contents_to_bash(layer,C,file_index):
    contents = [
        "#!/bin/bash\n",
        "#SBATCH -n 9\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-120:00\n",
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=40GB\n",
        f"#SBATCH -o out_train_fs_qt_resnet_layer{layer}_C{C}.out\n",
        f"#SBATCH -e out_train_fs_qt_resnet_layer{layer}_C{C}.err\n",
        "cd /public/home/hkust_jwliu_1/liuxi/Documents/pyCode/deep_field_collection/final_sum_qt_resnet\n",
        f"python3 -u train.py {num_epochs} {C} {layer}\n",
        ]
    out_chunk = outPath + f"/chunk{file_index // chunk_size}/"
    Path(out_chunk).mkdir(exist_ok=True, parents=True)
    outBashName = out_chunk + f"/train_qt_resnet_pbc_layer{layer}_C{C}.sh"
    with open(outBashName, "w+") as fptr:
        fptr.writelines(contents)

# Process each pair with its index
for file_index, layer_C in enumerate(layer_C_pairs_vec):
    layer,C=layer_C
    contents_to_bash(layer,C,file_index)