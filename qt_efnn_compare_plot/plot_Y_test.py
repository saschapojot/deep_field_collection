import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
from pathlib import Path
if (len(sys.argv) != 2):
    print("wrong number of arguments.")
    exit(21)


N = int(sys.argv[1])# Number of side
data_inDir=f"./train_test_data/N{N}/"

fileNameTest=data_inDir+"/db.test_num_samples40000.pkl"
with open(fileNameTest,"rb") as fptr:
    X_test, Y_test = pickle.load(fptr)

print(f"Data loaded. Y_test length: {len(Y_test)}")
# Calculate the average value
y_mean = np.mean(Y_test)
print(f"Average Y_test: {y_mean}")


textSize=25
legend_fontsize=23
fig_size=10
plt.figure(figsize=(fig_size, fig_size))
plt.hist(Y_test, bins=50, color='blue', alpha=0.7, edgecolor='black', label='Data Distribution')

plt.axvline(y_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {y_mean:.4f}')
plt.xlabel('Y_test Value',fontsize=textSize)
plt.ylabel('Number',fontsize=textSize)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="best", fontsize=legend_fontsize-8)
# plt.xticks(fontsize=xTickSize)
# plt.yticks(fontsize=yTickSize)
plt.tight_layout()
out_dir="./Y_test_figs/"
Path(out_dir).mkdir(exist_ok=True,parents=True)
plt.savefig(out_dir+f"/Y_test_N{N}_qt.png")
plt.savefig(out_dir+f"/Y_test_N{N}_qt.svg")
plt.close()

