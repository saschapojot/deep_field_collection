import numpy as np
import pickle
from structure import *

from pathlib import Path
#this function loads test data for larger lattices
#for all N, all layers
# Evaluation Function

def evaluate_model(model, test_loader, device):
    """
        Evaluate the trained model on the test dataset.

        Args:
            model (nn.Module): Trained model.
            test_loader (DataLoader): DataLoader for the test dataset.
            device (torch.device): Device to run evaluation on.

        Returns:
            float: Mean squared error on the test dataset.
        """

    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    criterion = torch.nn.MSELoss()  # Loss function
    all_predictions = []  # To store all predictions
    with torch.no_grad():  # No need to compute gradients during evaluation
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move data to device
            # Initialize S1 for the batch
            S1 = model.initialize_S1(X_batch)

            # Forward pass
            predictions = model(S1)
            all_predictions.append(predictions.cpu())

            # Compute loss
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item() * X_batch.size(0)  # Accumulate loss weighted by batch size

    # Compute average loss
    average_loss = total_loss / len(test_loader.dataset)
    all_predictions = torch.cat(all_predictions, dim=0)
    return average_loss, all_predictions  # ,custom_metric_sum,custom_metric_sum_another



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
N_for_model=10

N_vec=[10,15,20,25,30,35,40]
layer_all=[1,2,3]
C=9
set_epoch =1000#optimal is ?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_suffix=40000
for N in N_vec:
    for layer in layer_all:
        print(f"N={N}, layer={layer}==============================")
        in_model_dir = f"./out_model_data/N{N_for_model}/C{C}/layer{layer}/"
        in_model_file = in_model_dir + f"final_sum_qt_dnn_trained_over50_epoch{set_epoch}_num_samples200000.pth"
        out_Dir = f"./larger_lattice_test_performance/N{N}/"
        Path(out_Dir).mkdir(exist_ok=True, parents=True)
        checkpoint = torch.load(in_model_file, map_location=device)
        in_pkl_test_file = out_Dir + f"/db.test_num_samples{num_suffix}.pkl"
        with open(in_pkl_test_file, "rb") as fptr:
            X_test, Y_test = pickle.load(fptr)
        X_test_array = np.array(X_test)
        Y_test_array = np.array(Y_test)
        X_test_tensor = torch.tensor(X_test_array, dtype=torch.float)
        Y_test_tensor = torch.tensor(Y_test_array, dtype=torch.float)
        batch_size = 1000  # Define batch size
        # Create test dataset and DataLoader
        test_dataset = CustomDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False)  # Batch size can be adjusted

        # Load the trained model
        step_num_after_S1 = layer - 1
        model=final_sum_qt_dnn(
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
        num_params = count_parameters(model)
        print(f"num_params={num_params}")
        # Evaluate the model
        test_loss, predictions = evaluate_model(model, test_loader, device)
        std_loss = np.sqrt(test_loss)
        print(f"Test Loss (MSE): {test_loss:.6f}")
        predictions = np.array(predictions)
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        true_mean = np.mean(Y_test_array)
        print(f"pred_mean={pred_mean}, true_mean={true_mean}")
        outResultDir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer}/"
        Path(outResultDir).mkdir(parents=True, exist_ok=True)
        outTxtFile = outResultDir + f"/test_epoch{set_epoch}_num_samples{num_suffix}.txt"
        out_content = f"num_epochs={set_epoch}, MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}  N={N}, C={C}, layer={step_num_after_S1}, pred_mean={pred_mean}, pred_std={pred_std},  num_params={num_params}\n"
        with open(outTxtFile, "w+") as fptr:
            fptr.write(out_content)