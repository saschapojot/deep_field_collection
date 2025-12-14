import os
import re
import glob
from structure import *
import numpy as np
import pickle
import random
from datetime import datetime
#this script checks the time for computing 1 data point after training
#using EFNN
# Evaluation Function

# ==========================================
# 1. Configuration
# ==========================================
device = torch.device("cpu")
# Parameters
N_for_model = 10      # The N the model was trained on
N_test = 40         # The N of the data we are loading
layer = 3             # The layer depth
C = 3                 # Channels
set_epoch = 1000      # Epoch of the saved model
num_suffix = 40000    # Suffix for the test file

# Construct Paths
in_model_dir = f"./out_model_data/N{N_for_model}/C{C}/layer{layer}/"
in_model_file = in_model_dir + f"final_sum_qt_efnn_trained_over50_epoch{set_epoch}_num_samples200000.pth"

test_data_dir = f"./larger_lattice_test_performance/N{N_test}/"
in_pkl_test_file = test_data_dir + f"/db.test_num_samples{num_suffix}.pkl"

print(f"Loading data from: {in_pkl_test_file}")
print(f"Loading model from: {in_model_file}")

# ==========================================
# 2. Load Data & Select 1 Sample (Index 0)
# ==========================================
if not os.path.exists(in_pkl_test_file):
    print(f"Error: Data file not found at {in_pkl_test_file}")
    exit(1)

with open(in_pkl_test_file, "rb") as fptr:
    X_test, Y_test = pickle.load(fptr)


## Randomly select an index
ind_selected = random.randint(0, len(X_test) - 1)
print(f"Selected random index: {ind_selected} out of {len(X_test)}")

x_single = np.array(X_test[ind_selected])
y_single = np.array(Y_test[ind_selected])
# Convert to tensor and add batch dimension: Shape becomes [1, Channels, H, W]
x_tensor = torch.tensor(x_single, dtype=torch.float).unsqueeze(0).to(device)
y_tensor = torch.tensor(y_single, dtype=torch.float).unsqueeze(0).to(device)

print(f"Input Shape: {x_tensor.shape}")

# ==========================================
# 3. Load Model
# ==========================================
step_num_after_S1 = layer - 1
checkpoint = torch.load(in_model_file, map_location=device)
model = final_sum_qt_efnn(
    input_channels=3,
    phi0_out_channels=C,
    T_out_channels=C,
    nonlinear_conv1_out_channels=C,
    nonlinear_conv2_out_channels=C,
    final_out_channels=1,
    filter_size=filter_size,
    stepsAfterInit=step_num_after_S1
)
model.load_state_dict(checkpoint['model_state_dict'])  # Load saved weights
model.to(device)  # Move model to device
if not os.path.exists(in_model_file):
    print(f"Error: Model file not found at {in_model_file}")
    exit(1)


checkpoint = torch.load(in_model_file, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

# ==========================================
# 4. Compute Loop (Inference & Timing)
# ==========================================
total_time_seconds = 0.0
num_trials=17
print(f"\nStarting timing loop over {num_trials} samples...")
with torch.no_grad():
    for i in range(num_trials):
        # A. Randomly select an index
        ind_selected = random.randint(0, len(X_test) - 1)
        x_single = np.array(X_test[ind_selected])
        y_single = np.array(Y_test[ind_selected])
        # Convert to tensor and add batch dimension: Shape becomes [1, Channels, H, W]
        x_tensor = torch.tensor(x_single, dtype=torch.float).unsqueeze(0).to(device)
        y_tensor = torch.tensor(y_single, dtype=torch.float).unsqueeze(0).to(device)
        # B. Initialize S1 (Pre-computation)
        t_fw_start = datetime.now()
        S1 = model.initialize_S1(x_tensor)
        # C. Measure Time for Forward Pass
        prediction = model(x_tensor, S1)
        t_fw_end = datetime.now()
        # Accumulate time duration in seconds
        duration = (t_fw_end - t_fw_start).total_seconds()
        total_time_seconds += duration


# ==========================================
# 5. Output Results
# ==========================================
average_time = total_time_seconds / num_trials
print("-" * 30)
print(f"N={N_test}")
print(f"Number of trials: {num_trials}")
print(f"Total time:       {total_time_seconds:.6f} seconds")
print(f"Average time:     {average_time:.6e} seconds per sample")
print("-" * 30)