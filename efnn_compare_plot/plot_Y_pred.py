import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.ticker import MaxNLocator  # Import MaxNLocator

import pickle
from pathlib import Path

#this script plots Y_pred for efnn, 3-spin infinite-range model
L=15
K=455
r=3
layer_num=3

in_Y_pred_file=f"./efnn_data/out_model_L{L}_K{K}_r{r}_layer{layer_num}/inference_data.pkl"
with open(in_Y_pred_file,"rb") as fptr:
    X_test,Y_pred=pickle.load(fptr)

print(f"Data loaded. Y_pred length: {len(Y_pred)}")
# Calculate the average value
y_mean = np.mean(Y_pred)
print(f"Average Y_pred: {y_mean}")

# --- Modification Start: Calculate Custom Ticks ---
# 1. Find the min and max of the data to define the range
data_min = np.min(Y_pred)
data_max = np.max(Y_pred)

# 2. Generate 5 evenly spaced numbers between min and max
custom_ticks = np.linspace(data_min, data_max, 4)

# 3. Round them to the nearest integer and convert to integer type
custom_ticks = np.round(custom_ticks).astype(int)

# 4. Ensure values are unique (in case data range is very small, e.g., 0 to 2)
custom_ticks = np.unique(custom_ticks)
# --- Modification End ---
textSize=40
yTickSize=40
xTickSize=40
legend_fontsize=30
fig_size=10
plt.figure(figsize=(fig_size, fig_size))
ax = plt.gca()

plt.hist(Y_pred, bins=50, color='green', alpha=0.7, edgecolor='black', label='Y_pred')
plt.axvline(y_mean, color='magenta', linestyle='dashed', linewidth=2, label=f'Mean: {y_mean:.4f}')
plt.xlabel('Y_pred',fontsize=textSize)
plt.ylabel('Number',fontsize=textSize)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="best", fontsize=legend_fontsize, framealpha=0.5, markerfirst=False)

# --- Modification Start ---
# Force x-axis to use only integers
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# --- Modification End ---
plt.xticks(custom_ticks,fontsize=xTickSize)
plt.yticks(fontsize=yTickSize)
# --- Modification Start: Thicken the Box/Spines ---
spine_thickness = 3  # Increase this value for a thicker box
spine_color = 'black' # You can change this to 'black', 'navy', etc.
# Loop through all 4 spines (top, bottom, left, right) and adjust them
for spine in ax.spines.values():
    spine.set_linewidth(spine_thickness)
    spine.set_color(spine_color)
# --- Modification End ---
plt.tight_layout()
out_dir=f"./Y_pred_figs/"
Path(out_dir).mkdir(exist_ok=True,parents=True)
plt.savefig(out_dir+f"/Y_pred_L{L}_K_{K}_r{r}_layer{layer_num}.png")
plt.savefig(out_dir+f"/Y_pred_L{L}_K_{K}_r{r}_layer{layer_num}.svg")
plt.close()