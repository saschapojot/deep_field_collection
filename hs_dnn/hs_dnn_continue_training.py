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
from hs_dnn_structure import format_using_decimal, hs_dnn, CustomDataset

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    checkpoint_files = glob.glob(checkpoint_dir + "hs_dnn_trained_epoch*.pth")
    if not checkpoint_files:
        return None, 0

    # Sort by epoch number and get the latest
    checkpoint_files.sort(key=lambda x: int(x.split('epoch')[1].split('.pth')[0]))
    latest_checkpoint = checkpoint_files[-1]
    epoch_num = int(latest_checkpoint.split('epoch')[1].split('.pth')[0])

    return latest_checkpoint, epoch_num


def load_checkpoint_info(checkpoint_path, device):
    """Load checkpoint and extract information without applying states yet"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract learning rate from optimizer state
    optimizer_state = checkpoint['optimizer_state_dict']
    current_lr = optimizer_state['param_groups'][0]['lr']

    start_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']

    return checkpoint, start_epoch, last_loss, current_lr


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
additional_epochs = 10000

# Setup paths and parameters
B = list(combinations(range(L), r))
K = len(B)
data_inDir = f"./data_hs_L{L}_K_{K}_r{r}/"
fileNameTrain = data_inDir + "/hs20000.train.pkl"
out_model_Dir = f"./out_model_hs_efnn_L{L}_K{K}_r{r}_layer{num_layers}/neuron{num_neurons}/"

# Training parameters
batch_size = 1000
learning_rate_final = 1e-5  # Target final learning rate
weight_decay = 1e-4  # Match original training
save_interval = 1000
decrease_over = 400  # Match original training

# Check if model directory exists
if not Path(out_model_Dir).exists():
    print(f"Error: Model directory {out_model_Dir} does not exist!")
    print("Have you trained this model configuration before?")
    exit(1)


# Build checkpoint path from specified epoch
checkpoint_path = out_model_Dir + f"hs_dnn_trained_epoch{epoch_to_load}.pth"
if not Path(checkpoint_path).exists():
    print(f"Error: Checkpoint {checkpoint_path} does not exist!")
    # Show available checkpoints
    available_checkpoints = glob.glob(out_model_Dir + "hs_dnn_trained_epoch*.pth")
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
model = hs_dnn(
    num_spins=num_spins, num_layers=num_layers, num_neurons=num_neurons
).double().to(device)

# Load checkpoint info to get learning rate BEFORE creating optimizer
checkpoint, start_epoch, last_loss, current_lr = load_checkpoint_info(checkpoint_path, device)

print(f"Loaded checkpoint info:")
print(f"  Start epoch: {start_epoch}")
print(f"  Last loss: {last_loss:.6f}")
print(f"  Current learning rate: {current_lr:.8e}")

# Now load the model state
model.load_state_dict(checkpoint['model_state_dict'])
print("Model state loaded successfully")

# Create optimizer with the loaded learning rate
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay)

# Load optimizer state to restore momentum and other internal states
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("Optimizer state loaded successfully")

# Create scheduler for continued training with decay
# Calculate decay parameters for remaining training
if current_lr > learning_rate_final:
    # Calculate how many decay steps we need for the additional epochs
    step_of_decrease = additional_epochs // decrease_over
    if step_of_decrease > 0:
        # Calculate gamma to reach learning_rate_final from current_lr
        gamma = (learning_rate_final / current_lr) ** (1 / step_of_decrease)
        print(f"Continuing with decaying learning rate:")
        print(f"  decrease_over: {decrease_over}")
        print(f"  step_of_decrease: {step_of_decrease}")
        print(f"  gamma: {gamma:.8f}")
        print(f"  Expected final LR: {current_lr * (gamma ** step_of_decrease):.8e}")

        scheduler = StepLR(optimizer, step_size=decrease_over, gamma=gamma)
    else:
        # Not enough epochs for stepped decay, use exponential decay
        print(f"Not enough epochs for stepped decay, using exponential decay")
        decay_rate = (learning_rate_final / current_lr) ** (1 / additional_epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        print(f"  Exponential decay gamma: {decay_rate:.8f}")
else:
    print(f"Already at or below final learning rate ({learning_rate_final:.8e})")
    print(f"Continuing with constant learning rate: {current_lr:.8e}")
    scheduler = StepLR(optimizer, step_size=total_epochs + 1, gamma=1.0)  # No decay

# Load loss history
log_path = out_model_Dir + "training_log.txt"
loss_file_content = load_loss_history(log_path)
print(f"Loaded {len(loss_file_content)} lines of loss history")

# Verify current learning rate after all loading
verify_lr = optimizer.param_groups[0]['lr']
print(f"Verified learning rate after loading: {verify_lr:.8e}")

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

    # Get current learning rate before scheduler step
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {average_loss:.4f}")
    loss_file_content.append(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {average_loss:.8f}\n")

    # Update learning rate
    scheduler.step()

    # Get and print new learning rate after scheduler step
    new_lr = optimizer.param_groups[0]['lr']
    print(f"Learning Rate after Epoch {epoch + 1}: {new_lr:.8e}")

    # Save checkpoint
    if (epoch + 1) % save_interval == 0:
        save_path = out_model_Dir + f"hs_dnn_trained_epoch{epoch + 1}.pth"
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': average_loss,
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at Epoch {epoch + 1} to {save_path}")

        # Save updated log
        with open(log_path, "w") as f:
            f.writelines(loss_file_content)
        print(f"Loss log saved at Epoch {epoch + 1} to {log_path}")

# Save final results
with open(log_path, "w") as fptr:
    fptr.writelines(loss_file_content)

# Also save with the original naming convention for compatibility
with open(out_model_Dir + "hs_dnn_training_log.txt", "w") as fptr:
    fptr.writelines(loss_file_content)

# Save final model with consistent naming
final_model_path = out_model_Dir + f"hs_dnn_trained_epoch{total_epochs}.pth"
checkpoint = {
    'epoch': total_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': average_loss,
}
torch.save(checkpoint, final_model_path)
print(f"Final model saved to: {final_model_path}")

# Also save as simple model file for compatibility
torch.save(model.state_dict(), out_model_Dir + "hs_dnn_model.pth")

tTrainEnd = datetime.now()
print("Additional training time:", tTrainEnd - tTrainStart)
print(f"Training completed! Total epochs: {total_epochs}")
print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.8e}")