import torch
import torch.nn as nn
import sys
from itertools import combinations
import pickle

import numpy as np
import re
import glob
from hs_densenet_structure import *


L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)
num_layers=4
epoch_num=11999
growth_rate_dirs_vec=[]
growth_rate_vals_vec=[]

dirs_root=f"./out_model_hs_densenet_L{L}_K{K}_r{r}_layer{num_layers}/"

for dir in glob.glob(dirs_root+f"/growth_rate*"):
    match_grow_rate=re.search(r"growth_rate(\d+)", dir)
    if match_grow_rate:
        growth_rate_dirs_vec.append(dir)
        growth_rate_vals_vec.append(int(match_grow_rate.group(1)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



sorted_inds=np.argsort(growth_rate_vals_vec)
sorted_growth_rate_vec=[growth_rate_vals_vec[ind] for ind in sorted_inds]
sorted_growth_rate_dirs_vec=[growth_rate_dirs_vec[ind] for ind in sorted_inds]


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device="+str(device))
data_inDir=f"./data_hs_L{L}_K_{K}_r{r}/"
N_samples=int(20000)
fileNameTest=data_inDir+f"/hs{N_samples}.test.pkl"
with open(fileNameTest, 'rb') as f:
    X_test, Y_test = pickle.load(f)


# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1).to(device)

print(f"Y_test.shape={Y_test.shape}")

def load_one_model_performance(growth_rate,epoch_num,dir):
    model_file_name = dir + f"/hs_densenet_trained_epoch{epoch_num}.pth"
    # Create the model
    model = DenseNet_no_bn(
        num_spins=L, num_layers=num_layers, growth_rate=growth_rate
    )
    criterion = nn.MSELoss()
    # Load the checkpoint
    checkpoint = torch.load(model_file_name)
    # Extract the model state dictionary and load it
    model.load_state_dict(checkpoint['model_state_dict'])
    # Move the model to the appropriate device
    model = model.to(device)
    # Set the model to evaluation mode
    model.eval()
    print("Model successfully loaded and ready for evaluation.")
    errors = []
    with torch.no_grad():
        # Forward pass to get predictions
        predictions = model(X_test)
        # Compute loss or other evaluation metrics
        test_loss = criterion(predictions, Y_test).item()
        batch_errors = (predictions - Y_test).cpu().numpy()  # Convert to NumPy for easier handling
        errors.extend(batch_errors.flatten())  # Flatten and add to the list
    std_loss = np.sqrt(test_loss)
    num_params = count_parameters(model)
    print(f"Test Loss: {test_loss:.4f}, std loss: {std_loss:.4f}, num_params: {num_params}")
    outTxtFile = dir + f"/test_epoch{epoch_num}.txt"
    out_content = f"MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}\n"
    with open(outTxtFile, "w+") as fptr:
        fptr.write(out_content)
    print(f"processed {dir}")
# print(sorted_num_neurons_dirs_vec)
for ind in range(0,len(sorted_growth_rate_dirs_vec)):
    growth_rate=sorted_growth_rate_vec[ind]
    dir=sorted_growth_rate_dirs_vec[ind]
    load_one_model_performance(growth_rate,epoch_num, dir)