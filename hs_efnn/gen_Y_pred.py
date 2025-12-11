import pickle

import numpy as np
import re
import glob
from hs_efnn_structure import *

#this script generates Y_pred

L = 15  # Number of spins
r = 3   # Number of spins in each interaction term
B = list(combinations(range(L), r))
K=len(B)
num_layers=3
epoch_num=15999
neuron_num=150

dirs_root=f"./out_model_hs_efnn_L{L}_K{K}_r{r}_layer{num_layers}/neuron{neuron_num}/"
model_file_name=dirs_root+f"/hs_efnn_trained_epoch{epoch_num}.pth"
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
# Create the model
model=hs_efnn(
        num_spins=L,num_layers=num_layers,num_neurons=neuron_num
)
criterion = nn.MSELoss()
# Load the checkpoint
checkpoint = torch.load(model_file_name)
# Extract the model state dictionary and load it
model.load_state_dict(checkpoint['model_state_dict'])
# Move the model to the appropriate device
model = model.to(device)
# Set the model to evaluation mode
param_num = sum(p.numel() for p in model.parameters())
print(f"num_params={param_num}")
model.eval()
print("Model successfully loaded and ready for evaluation.")
with torch.no_grad():
    # Forward pass to get predictions
    predictions = model(X_test)
    Y_pred = predictions.cpu().numpy().flatten().tolist()

save_data = [X_test, Y_pred]
out_value_file=dirs_root+f"/inference_data_epoch{epoch_num}.pkl"
with open(out_value_file,"wb") as fptr:
    pickle.dump(save_data, fptr)