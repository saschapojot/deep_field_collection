import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pickle


if (len(sys.argv) != 3):
    print("wrong number of arguments.")
    exit(21)


L = int(sys.argv[1])# Number of spins
r = int(sys.argv[2]) # Number of spins in each interaction term



B = list(combinations(range(L), r))
K=len(B)
data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"

fileNameTest=data_inDir+"/inf_range.test.pkl"
with open(fileNameTest,"rb") as fptr:
    X_test, Y_test = pickle.load(fptr)

print(f"Data loaded. Y_test length: {len(Y_test)}")
# Calculate the average value
y_mean = np.mean(Y_test)
print(f"Average Y_test: {y_mean}")

width=6
height=8
textSize=25
yTickSize=33
xTickSize=33
legend_fontsize=23
marker_size1=100
marker_size2=80
lineWidth1=3
lineWidth2=2
tick_length=13
tick_width=2
minor_tick_length=7
minor_tick_width=1

# Plot histogram of Y_test
plt.figure(figsize=(width, height))
# plt.figure()
plt.hist(Y_test, bins=50, color='blue', alpha=0.7, edgecolor='black', label='Data Distribution')

# Add a vertical line for the average
plt.axvline(y_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {y_mean:.4f}')

# plt.title(f'Histogram of test set (Y_test) for classical 3-spin infinite range model')
plt.xlabel('Y_test Value',fontsize=textSize)
plt.ylabel('Number',fontsize=textSize)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="best", fontsize=legend_fontsize-8)
# plt.xticks(fontsize=xTickSize)
# plt.yticks(fontsize=yTickSize)
plt.tight_layout()
plt.savefig("Y_test_3_inf.png")
plt.savefig("Y_test_3_inf.svg")
plt.close()