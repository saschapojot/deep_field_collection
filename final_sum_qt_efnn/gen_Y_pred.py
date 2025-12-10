import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from structure import *

#this script computes Y_pred
def evaluate_model_get_predictions(model, test_loader, device):
    """
    Evaluate the trained model and return all predictions.
    Args:
        model:
        test_loader:
        device:

    Returns:

    """
    model.eval()  # Set model to evaluation mode
    all_predictions = []  # To store all predictions
    with torch.no_grad():  # No need to compute gradients
        for X_batch, _ in test_loader:  # We only need X to generate predictions
            X_batch = X_batch.to(device)
            # Initialize S1 for the batch
            S1 = model.initialize_S1(X_batch)
            # Forward pass
            predictions = model(X_batch, S1)
            all_predictions.append(predictions.cpu())
    # Concatenate all batch predictions into one tensor
    all_predictions = torch.cat(all_predictions, dim=0)
    return all_predictions

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# --- Configuration ---
N_for_model = 10
N_vec = [10, 15, 20, 25, 30, 35, 40]
layer_all = [1, 2, 3]
C = 3
set_epoch = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_suffix=40000

for N in N_vec:
    for layer in layer_all:
        print(f"Generating values for N={N}, layer={layer} ==============================")
        # 1. Define Paths
        # Source model path (Always trained on N=10)
        in_model_dir = f"./out_model_data/N{N_for_model}/C{C}/layer{layer}/"
        in_model_file = in_model_dir + f"final_sum_qt_efnn_trained_over50_epoch{set_epoch}_num_samples200000.pth"

        # Input Data path (Test data for current N)
        data_Dir = f"./larger_lattice_test_performance/N{N}/"
        in_pkl_test_file = data_Dir + f"/db.test_num_samples{num_suffix}.pkl"

        # Output Path for generated values
        outResultDir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer}/"
        Path(outResultDir).mkdir(parents=True, exist_ok=True)
        out_values_file = outResultDir + f"generated_values_epoch{set_epoch}.pkl"

        # 2. Load Data
        try:
            with open(in_pkl_test_file, "rb") as fptr:
                X_test, Y_test = pickle.load(fptr)
        except FileNotFoundError:
            print(f"Data file not found: {in_pkl_test_file}. Skipping...")
            continue

        X_test_array = np.array(X_test)
        Y_test_array = np.array(Y_test)

        X_test_tensor = torch.tensor(X_test_array, dtype=torch.float)
        Y_test_tensor = torch.tensor(Y_test_array, dtype=torch.float)

        batch_size = 1000
        test_dataset = CustomDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # 3. Load Model
        step_num_after_S1 = layer - 1
        # Ensure filter_size is available
        if 'filter_size' not in locals():
            print("Warning: filter_size variable not found. Assuming imported from structure or setting default.")
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
        try:
            checkpoint = torch.load(in_model_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print(f"Model file not found: {in_model_file}. Skipping...")
            continue
        model.to(device)
        # 4. Generate Predictions
        predictions_tensor = evaluate_model_get_predictions(model, test_loader, device)

        # Convert to numpy
        Y_pred = predictions_tensor.numpy()
        # 5. Save Results (X_test_array and Y_pred)
        # We create a list or tuple containing both arrays
        save_data = [X_test_array, Y_pred]
        with open(out_values_file, "wb") as f_out:
            pickle.dump(save_data, f_out)
        print(f"Saved X_test and Y_pred to: {out_values_file}")
        print(f"Shape of X_test: {X_test_array.shape}, Shape of Y_pred: {Y_pred.shape}")



print("All generation tasks completed.")