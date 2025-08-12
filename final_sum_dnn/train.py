
import sys
import pickle
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  DataLoader
from pathlib import Path
from itertools import combinations

from structure import *

if (len(sys.argv) != 5):
    print("wrong number of arguments.")
    exit(21)

L = int(sys.argv[1])# Number of spins
r = int(sys.argv[2]) # Number of spins in each interaction term
num_layers = int(sys.argv[3])
num_neurons=int(sys.argv[4])
B = list(combinations(range(L), r))
K=len(B)
data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"

fileNameTrain=data_inDir+"/inf_range.train.pkl"

batch_size = 1000
learning_rate = 0.001
learning_rate_final=1e-5

epoch_multiple=1000
weight_decay = 1e-3  # L2 regularization strength

with open(fileNameTrain,"rb") as fptr:
    X_train, Y_train = pickle.load(fptr)

num_sample,num_spins=X_train.shape
print(f"num_sample={num_sample}")
num_epochs =int(num_sample/batch_size*epoch_multiple)
print(f"num_epochs={num_epochs}")
save_interval=1000
decrease_over=400
step_of_decrease = num_epochs // decrease_over
print(f"step_of_decrease={step_of_decrease}")
gamma = (learning_rate_final/learning_rate) ** (1/step_of_decrease)

print(f"gamma={gamma}")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device="+str(device))

X_train = torch.tensor(X_train, dtype=torch.float64).to(device)  # Move to device
Y_train = torch.tensor(Y_train, dtype=torch.float64).view(-1, 1).to(device)

# Create Dataset and DataLoader
dataset = CustomDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model=dnn_final_sum(
num_spins=num_spins,num_layers=num_layers,num_neurons=num_neurons
).double().to(device)

param_num=sum(p.numel() for p in model.parameters())
print(f"param_num={param_num}")

# Define loss function and optimizer with L2 regularization
# Optimizer and loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every step_size epochs
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=gamma)


out_model_Dir=f"./out_model_final_sum_dnn_L{L}_K{K}_r{r}_layer{num_layers}/neuron{num_neurons}/"
Path(out_model_Dir).mkdir(exist_ok=True,parents=True)
loss_file_content=[]
print(f"num_spin={num_spins}")

# Training loop
tTrainStart = datetime.now()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move batch to device
        nRow, _ = X_batch.shape
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
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    loss_file_content.append(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.8f}\n")
    # Update the learning rate
    scheduler.step()
    # Optionally print the current learning rate
    current_lr = scheduler.get_last_lr()[0]
    print(f"Learning Rate after Epoch {epoch + 1}: {current_lr:.8e}")
    if (epoch + 1) % save_interval == 0:
        save_path = out_model_Dir + f"/final_sum_dnn_trained_epoch{epoch}.pth"
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
        log_path = out_model_Dir + f"/training_log.txt"
        with open(log_path, "w") as f:
            f.writelines(loss_file_content)
        print(f"Loss log saved at Epoch {epoch + 1} to {log_path}")


# Save the loss log
with open(out_model_Dir + f"/training_log.txt", "w+") as fptr:
    fptr.writelines(loss_file_content)

# Save the model
torch.save(model.state_dict(), out_model_Dir + f"final_sum_dnn_model.pth")
tTrainEnd = datetime.now()
print("Training time:", tTrainEnd - tTrainStart)