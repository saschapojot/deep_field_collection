import torch
import torch.nn as nn
import sys
from itertools import combinations
import pickle

import numpy as np
import re
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
from resnet_structure_final_sum import format_using_decimal, resnet_final_sum,CustomDataset

# if (len(sys.argv) != 3):
#     print("wrong number of arguments.")
#     exit(21)
#
# L = int(sys.argv[1])# Number of spins
# r = int(sys.argv[2]) # Number of spins in each interaction term
L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)
num_layers=3
num_neurons_vec=[]
num_neurons_dirs_vec=[]
dirs_root=f"./out_model_resnet_final_sum_L{L}_K{K}_r{r}_layer{num_layers}/"

for dir in glob.glob(dirs_root+f"/neuron*"):
    match_neuron_num=re.search(r"neuron(\d+)", dir)
    if match_neuron_num:
        num_neurons_dirs_vec.append(dir)
        num_neurons_vec.append(int(match_neuron_num.group(1)))


sorted_inds=np.argsort(num_neurons_vec)
sorted_num_neurons_vec=[num_neurons_vec[ind] for ind in sorted_inds]
sorted_num_neurons_dirs_vec=[num_neurons_dirs_vec[ind] for ind in sorted_inds]
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device="+str(device))
data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
fileNameTest=data_inDir+"/inf_range.test.pkl"
with open(fileNameTest, 'rb') as f:
    X_test, Y_test = pickle.load(f)


# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1).to(device)

print(f"Y_test.shape={Y_test.shape}")


def load_one_model_performance(num_neurons,dir):
    model_file_name=dir+"/resnet_final_sum_model.pth"
    # Set model to evaluation mode
    model=resnet_final_sum(num_spins=L,num_layers=num_layers,num_neurons=num_neurons)
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load(model_file_name))
    # Move the model to the appropriate device
    model = model.to(device)
    # Set the model to evaluation mode
    model.eval()

    print("Model successfully loaded and ready for evaluation.")
    # Disable gradient computation for evaluation
    errors = []
    with torch.no_grad():
        # Forward pass to get predictions
        predictions = model(X_test)
        # Compute loss or other evaluation metrics
        test_loss = criterion(predictions, Y_test).item()
        batch_errors = (predictions - Y_test).cpu().numpy()  # Convert to NumPy for easier handling
        errors.extend(batch_errors.flatten())  # Flatten and add to the list
    # print(errors)
    print(f"Test Loss: {test_loss:.4f}")
    # Convert errors to a NumPy array
    # errors = np.array(errors)
    std_loss = np.sqrt(test_loss)
    outTxtFile=dir+"/test.txt"
    out_content = f"MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}\n"
    with open(outTxtFile, "w+") as fptr:
        fptr.write(out_content)
    print(f"processed {dir}")

for ind in range(0,len(sorted_num_neurons_dirs_vec)):
    num_neurons=sorted_num_neurons_vec[ind]
    dir=sorted_num_neurons_dirs_vec[ind]
    load_one_model_performance(num_neurons, dir)
