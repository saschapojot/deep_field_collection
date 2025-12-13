import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


in_csv_file="./compare_time.csv"

df = pd.read_csv(in_csv_file,index_col=0)

df.index.name = None
# Get the columns directly
N_values = df.columns
N_vec=[int(elem) for elem in N_values]

efnn_time_vec=np.array(df.iloc[0,:])

ed_time_vec=np.array(df.iloc[1,:])

textSize=40
yTickSize=40
xTickSize=40
legend_fontsize=30
marker_size2=80
fig_size=10
lineWidth2=2
plt.figure(figsize=(fig_size, fig_size))
ax = plt.gca()

#plot efnn time
plt.scatter(N_vec,efnn_time_vec,color="limegreen", label="EFNN",s=marker_size2)
plt.plot(N_vec,efnn_time_vec,color="limegreen", linestyle="dashed",linewidth=lineWidth2)

#plot ed time
plt.scatter(N_vec,ed_time_vec,color="red",label="ED",s=marker_size2,marker="s")
plt.plot(N_vec,ed_time_vec,color="red",linestyle="dashed",linewidth=lineWidth2)
plt.yscale('log')  # Consider log scale if ranges vary widely
plt.xticks(fontsize=xTickSize)
plt.yticks(fontsize=yTickSize)
plt.xlabel("$N$",fontsize=textSize)
plt.ylabel("time/s",fontsize=textSize)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="best", fontsize=legend_fontsize, framealpha=0.5, markerfirst=False)

# --- Modification Start: Thicken the Box/Spines ---
spine_thickness = 3  # Increase this value for a thicker box
spine_color = 'black' # You can change this to 'black', 'navy', etc.
# Loop through all 4 spines (top, bottom, left, right) and adjust them
for spine in ax.spines.values():
    spine.set_linewidth(spine_thickness)
    spine.set_color(spine_color)
# --- Modification End ---
plt.tight_layout()

plt.savefig("efnn_ed_time.png")
plt.savefig("efnn_ed_time.svg")
plt.close()



log_ed_time = np.log(ed_time_vec)

# Perform linear regression (degree 1 polynomial)
slope, intercept = np.polyfit(np.log(N_vec), log_ed_time, 1)


print(f"Regression Results (Power Law Fit):")
print(f"Slope (Exponent p): {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"Inferred Complexity: O(N^{slope:.2f})")

