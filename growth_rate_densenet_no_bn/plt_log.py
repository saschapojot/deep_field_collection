import numpy as np
import re
import matplotlib.pyplot as plt
import glob
from itertools import combinations

# System Parameters
L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)

# num_skip=3
num_layers=3
num_neurons_vec=[]
num_neurons_dirs_vec=[]
name="densenet no bn"
dirs_root=f"./out_model_densenet_no_bn_L{L}_K{K}_r{r}_layer{num_layers}/"

for dir in glob.glob(dirs_root+f"/neuron*"):
    match_neuron_num=re.search(r"neuron(\d+)", dir)
    if match_neuron_num:
        num_neurons_dirs_vec.append(dir)
        num_neurons_vec.append(int(match_neuron_num.group(1)))


sorted_inds=np.argsort(num_neurons_vec)
sorted_num_neurons_vec=[num_neurons_vec[ind] for ind in sorted_inds]
sorted_num_neurons_dirs_vec=[num_neurons_dirs_vec[ind] for ind in sorted_inds]

def load_one_log(dir):
    log_fileName=dir+"/growth_rate_densenet_no_bn_training_log.txt"
    with open(log_fileName, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    pattern_float = r'Loss:\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'

    loss_vec = []
    for one_line in lines:
        match_loss = re.search(pattern_float, one_line)
        if match_loss:
            loss_vec.append(float(match_loss.group(1)))
        else:
            print("format error")
            exit(12)
    return  loss_vec


for ind in range(0,len(sorted_num_neurons_dirs_vec)):
    dir = sorted_num_neurons_dirs_vec[ind]
    num_neurons = sorted_num_neurons_vec[ind]
    loss_vec = load_one_log(dir)
    plt.figure(figsize=(10, 6))
    epoch_vec = list(range(0, len(loss_vec)))
    truncate_at = 0
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_vec[truncate_at:], loss_vec[truncate_at:], label="Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    y_intersection = loss_vec[-1]
    plt.axhline(y=y_intersection, color='r', linestyle='--', label=f'{y_intersection}')
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"{name}: Training Loss Over Epochs, layer={num_layers}, neurons={num_neurons}", fontsize=14)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.savefig(dir + "/loss.png")
    plt.close()
    print(f"processed {dir}")