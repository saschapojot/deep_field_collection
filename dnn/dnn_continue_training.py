import torch
import torch.nn as nn
import sys
from itertools import combinations
import pickle
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
from dnn_structure import format_using_decimal, dnn, CustomDataset


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    checkpoint_files = glob.glob(checkpoint_dir + "hs_efnn_trained_epoch*.pth")
    if not checkpoint_files:
        return None, 0

    # Sort by epoch number and get the latest
    checkpoint_files.sort(key=lambda x: int(x.split('epoch')[1].split('.pth')[0]))
    latest_checkpoint = checkpoint_files[-1]
    epoch_num = int(latest_checkpoint.split('epoch')[1].split('.pth')[0])

    return latest_checkpoint, epoch_num


def load_training_state(checkpoint_path, model, optimizer, scheduler, device):
    """Load model, optimizer, and scheduler states from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']

    return start_epoch, last_loss


def load_loss_history(log_path):
    """Load previous loss history from log file"""
    if Path(log_path).exists():
        with open(log_path, 'r') as f:
            return f.readlines()
    return []
# Main script starts here
if len(sys.argv) != 6:
    print("Usage: python continue_training.py L r num_layers num_neurons epoch_to_load")
    print("Example: python continue_training.py 15 3 1 90 9999")
    exit(1)

# Parse arguments
L = int(sys.argv[1])
r = int(sys.argv[2])
num_layers = int(sys.argv[3])
num_neurons = int(sys.argv[4])
epoch_to_load = int(sys.argv[5])

# Fixed additional epochs
additional_epochs = 16000
# Setup paths and parameters
B = list(combinations(range(L), r))
K = len(B)
data_inDir = f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
fileNameTrain = data_inDir + "/inf_range.train.pkl"
out_model_Dir = f"./out_model_dnn_L{L}_K{K}_r{r}_layer{num_layers}/neuron{num_neurons}/"
# Training parameters
batch_size = 1000
learning_rate = 1e-5
learning_rate_final = 1e-6
weight_decay = 1e-3
save_interval = 1000

# Check if model directory exists
if not Path(out_model_Dir).exists():
    print(f"Error: Model directory {out_model_Dir} does not exist!")
    print("Have you trained this model configuration before?")
    exit(1)

# Build checkpoint path from specified epoch
checkpoint_path = out_model_Dir + f"hs_efnn_trained_epoch{epoch_to_load}.pth"
if not Path(checkpoint_path).exists():
    print(f"Error: Checkpoint {checkpoint_path} does not exist!")
    # Show available checkpoints
    available_checkpoints = glob.glob(out_model_Dir + "hs_efnn_trained_epoch*.pth")
    if available_checkpoints:
        available_epochs = sorted([int(cp.split('epoch')[1].split('.pth')[0]) for cp in available_checkpoints])
        print(f"Available checkpoint epochs: {available_epochs}")
    exit(1)

print(f"Loading checkpoint from epoch {epoch_to_load}")
# Load data
print("Loading training data...")
with open(fileNameTrain, "rb") as fptr:
    X_train, Y_train = pickle.load(fptr)

num_sample, num_spins = X_train.shape
print(f"num_sample={num_sample}")
# Calculate total epochs
epoch_multiple = 1000
original_num_epochs = int(num_sample / batch_size * epoch_multiple)
total_epochs = epoch_to_load + additional_epochs
print(f"Original training epochs: {original_num_epochs}")
print(f"Loading from epoch: {epoch_to_load}")
print(f"Training for {additional_epochs} additional epochs")
print(f"Total epochs will be: {total_epochs}")


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# Prepare data
X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float64).view(-1, 1).to(device)

dataset = CustomDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model
model = dnn(
    num_spins=num_spins, num_layers=num_layers, num_neurons=num_neurons
).double().to(device)
# Setup optimizer and scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# For continuing training, we need to adjust the scheduler
# Calculate the original scheduler parameters
decrease_over = 400
step_of_decrease = original_num_epochs // decrease_over
gamma = (learning_rate_final / learning_rate) ** (1 / step_of_decrease)

scheduler = StepLR(optimizer, step_size=decrease_over, gamma=gamma)

# Load checkpoint
start_epoch, last_loss = load_training_state(checkpoint_path, model, optimizer, scheduler, device)

# Load loss history
log_path = out_model_Dir + "dnn_training_log.txt"
loss_file_content = load_loss_history(log_path)
print(f"Loaded {len(loss_file_content)} lines of loss history")
print(f"Last loss: {last_loss:.6f}")
# Get current learning rate
current_lr = scheduler.get_last_lr()[0]
print(f"Current learning rate: {current_lr:.8e}")

# Training loop
print(f"\nContinuing training from epoch {start_epoch} to {total_epochs}...")
tTrainStart = datetime.now()

for epoch in range(start_epoch, total_epochs):
    model.train()
    epoch_loss = 0

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        predictions = model(X_batch)

        # Compute loss
        loss = criterion(predictions, Y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate batch loss
        epoch_loss += loss.item() * X_batch.size(0)

    # Average loss over total samples
    average_loss = epoch_loss / len(dataset)
    print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {average_loss:.4f}")
    loss_file_content.append(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {average_loss:.8f}\n")

    # Update learning rate
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Learning Rate after Epoch {epoch + 1}: {current_lr:.8e}")

    # Save checkpoint
    if (epoch + 1) % save_interval == 0:
        save_path = out_model_Dir + f"hs_efnn_trained_epoch{epoch}.pth"
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': average_loss,
        }
        torch.save(checkpoint, save_path)

        # Save updated log
        with open(log_path, "w") as f:
            f.writelines(loss_file_content)
        print(f"Checkpoint and log saved at Epoch {epoch + 1}")

# Save final results
with open(log_path, "w") as fptr:
    fptr.writelines(loss_file_content)

# Save final model
final_model_path = out_model_Dir + f"dnn_model_continued_epoch{total_epochs}.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to: {final_model_path}")

tTrainEnd = datetime.now()
print("Additional training time:", tTrainEnd - tTrainStart)
print(f"Training completed! Total epochs: {total_epochs}")