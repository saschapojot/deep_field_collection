import pickle
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from qt_vit_structure import *
import sys
from datetime import datetime

argErrCode=3

if (len(sys.argv)!=5):
    print("wrong number of arguments")
    print("example: qt_vit_train.py num_epochs C num_layers D")
    exit(argErrCode)


num_epochs = int(sys.argv[1])
learning_rate = 1e-3
learning_rate_final=1e-4
weight_decay = 1e-5
N=10
decrease_over = 50
C=int(sys.argv[2])
num_layers=int(sys.argv[3])
D=int(sys.argv[4])
inDir=f"./train_test_data/N{N}/"

num_samples_in=200000
in_pkl_train_file=inDir+f"/db.train_num_samples{num_samples_in}.pkl"

step_of_decrease = num_epochs // decrease_over
print(f"step_of_decrease={step_of_decrease}")
gamma = (learning_rate_final/learning_rate) ** (1/step_of_decrease)

print(f"gamma={gamma}")
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device="+str(device))

with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)


# Convert to NumPy arrays

X_train_array = np.array(X_train)  # Shape: (num_samples, 3,N, N, )
Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

# Convert NumPy arrays to PyTorch tensors with dtype=torch.float64
X_train_tensor = torch.tensor(X_train_array, dtype=torch.float)  # Shape: (num_samples, 3, N, N)
Y_train_tensor = torch.tensor(Y_train_array, dtype=torch.float)  # Shape: (num_samples,)

# Instantiate the dataset
train_dataset = CustomDataset(X_train_tensor, Y_train_tensor)
# Create DataLoader for training
batch_size = 1000 # Define batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model=qt_vit(
input_channels=3,
T0_out_channels=C,
layer_num=num_layers,
embed_dim=D
).to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
num_params=count_parameters(model)
print(f"num_params={num_params}")
# Optimizer, scheduler, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every step_size epochs
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=gamma)
criterion = nn.MSELoss()
out_model_dir=f"./out_model_data/N{N}/C{C}/layer{num_layers}/D{D}/"
Path(out_model_dir).mkdir(exist_ok=True,parents=True)

tStart=datetime.now()
# To log loss values for each epoch
loss_file_content = []
# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0  # Reset epoch loss
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move batch to device
        predictions = model(X_batch)
        # Compute loss
        loss = criterion(predictions, Y_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accumulate batch loss (scaled by batch size)
        epoch_loss += loss.item() * X_batch.size(0)
    # Compute average loss over all samples
    average_loss = epoch_loss / len(train_dataset)
    # Log epoch summary
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    loss_file_content.append(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.8f}\n")
    # Update the learning rate
    scheduler.step()
    # Optionally log the current learning rate
    current_lr = scheduler.get_last_lr()[0]
    print(f"Learning Rate after Epoch {epoch + 1}: {current_lr:.8e}")
    if (epoch + 1) % save_interval == 0:
        save_suffix = f"_over{decrease_over}_epoch{epoch + 1}_num_samples{num_samples_in}"
        save_path = out_model_dir + f"/qt_densenet_trained{save_suffix}.pth"
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': average_loss,
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at Epoch {epoch + 1} to {save_path}")
        # Save the current loss logs at checkpoint intervals
        log_path = out_model_dir + f"/training_log{save_suffix}.txt"
        with open(log_path, "w") as f:
            f.writelines(loss_file_content)
        print(f"Loss log saved at Epoch {epoch + 1} to {log_path}")

suffix_str=f"_over{decrease_over}_num_samples{num_samples_in}"
# Save training log to file
with open(out_model_dir+f"/training_log{suffix_str}.txt", "w") as f:
    f.writelines(loss_file_content)

# Save the trained model
# torch.save(model.state_dict(), out_model_dir+f"/dsnn_qt_trained{suffix_str}.pth")
print("Training complete")

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")