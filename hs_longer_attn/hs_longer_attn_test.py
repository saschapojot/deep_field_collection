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

from hs_longer_attn_structure import  *

L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)


num_layers=3
epoch_num=15999
num_heads=5

num_embed_dim_vec=[]
num_embed_dim_dirs_vec=[]
dirs_root=f"./out_model_hs_attn_L{L}_K{K}_r3_layer{num_layers}/"

for dir in glob.glob(dirs_root+f"/embed_dim*"):
    match_neuron_num=re.search(r"embed_dim(\d+)", dir)
    if match_neuron_num:
        num_embed_dim_dirs_vec.append(dir)
        num_embed_dim_vec.append(int(match_neuron_num.group(1)))



sorted_inds=np.argsort(num_embed_dim_vec)
sorted_num_embed_dim_vec=[num_embed_dim_vec[ind] for ind in sorted_inds]
sorted_num_embed_dim_dirs_vec=[num_embed_dim_dirs_vec[ind] for ind in sorted_inds]
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
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def load_one_model_performance(embed_dim,dir):
    model_file_name = dir + "/hs_efnn_trained_epoch1999.pth"
    # Create the model
    model=hs_longer_attn(
        num_spins=L,num_layers=num_layers,num_heads=num_heads,embed_dim=embed_dim
    )
    criterion = nn.MSELoss()
    # Load the checkpoint
    checkpoint = torch.load(model_file_name)
    # Extract the model state dictionary and load it
    model.load_state_dict(checkpoint['model_state_dict'])
    # Move the model to the appropriate device
    model = model.to(device)
    # Set the model to evaluation mode
    # param_num = sum(p.numel() for p in model.parameters())
    # print(f"param_num={param_num}")
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


for ind in range(0,len(sorted_num_embed_dim_vec)):
    embed_dim=sorted_num_embed_dim_vec[ind]
    dir=sorted_num_embed_dim_dirs_vec[ind]
    load_one_model_performance(embed_dim,dir)