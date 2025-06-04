from qt_resnet_structure import *
import sys
import pickle
import numpy as np
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from datetime import datetime

argErrCode=3
if (len(sys.argv)!=4):
    print("wrong number of arguments")
    print("example: python qt_resnet_train.py num_epochs C step_num_after_S1")
    exit(argErrCode)
num_epochs = int(sys.argv[1])
learning_rate = 1e-3
learning_rate_final=1e-4
weight_decay = 1e-5
N=10
decrease_over = 50

C=int(sys.argv[2])
step_num_after_S1=int(sys.argv[3])


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


# Instantiate the network

model=qt_resnet(
  input_channels=3,
    phi0_out_channels=C,
    T_out_channels=C,
    nonlinear_conv1_out_channels=C,
    nonlinear_conv2_out_channels=C,
    final_out_channels=1,
    filter_size=filter_size,
stepsAfterInit=step_num_after_S1
).to(device)

# Optimizer, scheduler, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every step_size epochs
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=gamma)
criterion = nn.MSELoss()
out_model_dir=f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1+1}/"
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

        # Initialize S1 for the batch
        S1 = model.initialize_S1(X_batch)

        # Forward pass
        predictions = model(S1)

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
        save_path = out_model_dir + f"/dsnn_qt_trained{save_suffix}.pth"
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


suffix_str=f"_over{decrease_over}_epoch{num_epochs}_num_samples{num_samples_in}"
# Save training log to file
with open(out_model_dir+f"/training_log{suffix_str}.txt", "w") as f:
    f.writelines(loss_file_content)


# Save the trained model
# torch.save(model.state_dict(), out_model_dir+f"/dsnn_qt_trained{suffix_str}.pth")
print("Training complete")

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")