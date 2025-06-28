import os
import re
import glob
import numpy as np
import pickle
from qt_vit_structure import *
#this script compute performance on test set
#using models saved at different epochs

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
    with torch.no_grad():  # No need to compute gradients during evaluation
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move data to device
            predictions = model(X_batch)
            # Compute loss
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item() * X_batch.size(0)  # Accumulate loss weighted by batch size

    # Compute average loss
    average_loss = total_loss / len(test_loader.dataset)
    return average_loss

N=10
C=3
layer=4
D=40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_model_dir=f"./out_model_data/N{N}/C{C}/layer{layer}/D{D}/"

pth_file_vec=[]
for pthFile in glob.glob(os.path.join(in_model_dir, "*.pth")):
    pth_file_vec.append(pthFile)
in_pkl_Dir=f"./train_test_data/N{N}/"
in_pkl_test_file=in_pkl_Dir+"/db.test_num_samples200000.pkl"
# print(f"pth_file_vec={pth_file_vec}")
with open(in_pkl_test_file,"rb") as fptr:
    X_test, Y_test = pickle.load(fptr)
X_test_array=np.array(X_test)
Y_test_array=np.array(Y_test)
print(f"Y_test_array.shape={Y_test_array.shape}")
X_test_tensor=torch.tensor(X_test_array,dtype=torch.float)

Y_test_tensor=torch.tensor(Y_test_array,dtype=torch.float)
batch_size = 1000  # Define batch size

# Create test dataset and DataLoader
test_dataset = CustomDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)  # Batch size can be adjusted
pattern_epoch=r'epoch(\d+)'

def extract_epoch(pth_file):
    match_epoch=re.search(pattern_epoch,pth_file)
    if match_epoch:
        return int(match_epoch.group(1))
    else:
        print(f"format error: {pth_file}")
        exit(12)

epoch_vals=[extract_epoch(file) for file in pth_file_vec]
inds_epoch_vals=np.argsort(epoch_vals)
sorted_pth_file_vec=[pth_file_vec[j] for j in inds_epoch_vals]
sorted_epoch_vec=[epoch_vals[j] for j in inds_epoch_vals]
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_epoch_test(in_pth_file,test_loader, device):
    # Load the trained model
    checkpoint = torch.load(in_pth_file, map_location=device)
    model=qt_vit(
        input_channels=3,T0_out_channels=C,layer_num=layer,embed_dim=D
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move model to device
    test_loss = evaluate_model(model, test_loader, device)
    std_loss = np.sqrt(test_loss)
    num_params = count_parameters(model)
    return test_loss, std_loss,num_params

def one_line_outFile(epochNum,test_loss,std_loss,N,C,layer,num_params):
    msg = f"num_epochs={epochNum}, MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}  N={N}, C={C}, layer={layer}, num_params={num_params}\n"
    return msg


#contents in output file
outLineAll=[]
outTextFileName=in_model_dir+"/test_over_epochs.txt"
for j in range(0,len(sorted_pth_file_vec)):
    epoch = sorted_epoch_vec[j]
    pth_file = sorted_pth_file_vec[j]
    test_loss, std_loss, num_params=one_epoch_test(pth_file, test_loader, device)
    print(f"epoch={epoch}, std_loss={std_loss}, num_params={num_params}")
    oneLine = one_line_outFile(epoch, test_loss, std_loss, N, C, layer,num_params)
    outLineAll.append(oneLine)


with open(outTextFileName,"w+") as fptr:
    fptr.writelines(outLineAll)