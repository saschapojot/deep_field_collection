import pickle
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt  # Added for creating the histogram

#this script shows statistics of data


L = 15# Number of spins
r = 3 # Number of spins in each interaction term
B = list(combinations(range(L), r))
print(f"len(B)={len(B)}")
K=len(B)

inDir=f"./data_hs_L{L}_K_{K}_r{r}/"
#save training data
fileNameTrain=inDir+"/hs.train.pkl"
with open(fileNameTrain,"rb") as fptr:
    X_train, Y_train = pickle.load(fptr)
Y_train_mean=np.mean(Y_train)
print(f"len(Y_train)={len(Y_train)}, Y_train_mean={Y_train_mean}")
# Create a histogram of Y_train data
plt.figure(figsize=(10, 6))
plt.hist(Y_train, bins=300, alpha=0.7, color='blue', edgecolor='black')
plt.title(f'Histogram of Y_train Data (L={L}, K={K}, r={r}), mean={Y_train_mean:.4f}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hs_stats.png")
plt.close()