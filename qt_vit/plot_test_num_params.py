import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from itertools import combinations

# plots test result and parameter number

root_dir="./out_model_data/"
layer_num_vec=[1,2,3,4]
epoch_num=700

qt_vit_inCsvName_layer0=root_dir+f"/layer{layer_num_vec[0]}_epoch{epoch_num}_std_loss.csv"
qt_vit_inCsvName_layer1=root_dir+f"/layer{layer_num_vec[1]}_epoch{epoch_num}_std_loss.csv"
qt_vit_inCsvName_layer2=root_dir+f"/layer{layer_num_vec[2]}_epoch{epoch_num}_std_loss.csv"
qt_vit_inCsvName_layer3=root_dir+f"/layer{layer_num_vec[3]}_epoch{epoch_num}_std_loss.csv"

qt_vit_in_df_layer0=pd.read_csv(qt_vit_inCsvName_layer0)
qt_vit_in_df_layer1=pd.read_csv(qt_vit_inCsvName_layer1)
qt_vit_in_df_layer2=pd.read_csv(qt_vit_inCsvName_layer2)
qt_vit_in_df_layer3=pd.read_csv(qt_vit_inCsvName_layer3)

#qt_vit, layer 1
layer1=layer_num_vec[0]
qt_vit_D_vec_layer0=np.array(qt_vit_in_df_layer0["D"])
qt_vit_std_loss_vec_layer0=np.array(qt_vit_in_df_layer0["std_loss"])
qt_vit_num_params_vec_layer0=np.array(qt_vit_in_df_layer0["num_params"])

#qt_vit, layer 2
layer2=layer_num_vec[1]
qt_vit_D_vec_layer1=np.array(qt_vit_in_df_layer1["D"])
qt_vit_std_loss_vec_layer1=np.array(qt_vit_in_df_layer1["std_loss"])
qt_vit_num_params_vec_layer1=np.array(qt_vit_in_df_layer1["num_params"])

#qt_vit, layer 3
layer3=layer_num_vec[2]
qt_vit_D_vec_layer2=np.array(qt_vit_in_df_layer2["D"])
qt_vit_std_loss_vec_layer2=np.array(qt_vit_in_df_layer2["std_loss"])
qt_vit_num_params_vec_layer2=np.array(qt_vit_in_df_layer2["num_params"])


#qt_vit, layer 4
layer4=layer_num_vec[3]
qt_vit_D_vec_layer3=np.array(qt_vit_in_df_layer3["D"])
qt_vit_std_loss_vec_layer3=np.array(qt_vit_in_df_layer3["std_loss"])
qt_vit_num_params_vec_layer3=np.array(qt_vit_in_df_layer3["num_params"])

# width=6
# height=8
# textSize=33
# yTickSize=33
# xTickSize=33
# legend_fontsize=20
lineWidth1=1
marker_size1=80
# tick_length=13
# tick_width=2
# minor_tick_length=7
# minor_tick_width=1
#num parameters
plt.figure()
plt.minorticks_on()
plt.scatter(qt_vit_D_vec_layer0,qt_vit_num_params_vec_layer0,color="blue",marker="o",s=marker_size1,label=f"ViT, n={layer1}")
plt.plot(qt_vit_D_vec_layer0,qt_vit_num_params_vec_layer0,color="blue",linestyle="dashed",linewidth=lineWidth1)

plt.scatter(qt_vit_D_vec_layer1,qt_vit_num_params_vec_layer1,color="magenta",marker="^",s=marker_size1,label=f"ViT, n={layer2}")
plt.plot(qt_vit_D_vec_layer1,qt_vit_num_params_vec_layer1,color="magenta",linestyle="dashed",linewidth=lineWidth1)

plt.scatter(qt_vit_D_vec_layer2,qt_vit_num_params_vec_layer2,color="green",marker="s",s=marker_size1,label=f"ViT, n={layer3}")
plt.plot(qt_vit_D_vec_layer2,qt_vit_num_params_vec_layer2,color="green",linestyle="dashed",linewidth=lineWidth1)

plt.scatter(qt_vit_D_vec_layer3,qt_vit_num_params_vec_layer3,color="red",marker="8",s=marker_size1,label=f"ViT, n={layer4}")
plt.plot(qt_vit_D_vec_layer3,qt_vit_num_params_vec_layer3,color="red",linestyle="dashed",linewidth=lineWidth1)
x_font_size=25
y_font_size=25
title_size=20
plt.xlabel("$D$",fontsize=x_font_size)
plt.ylabel("Number of Parameters",fontsize=y_font_size)
plt.title(f"ViT",fontsize=title_size)
plt.yscale("log")  # If your errors span multiple orders of magnitude
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.yticks(fontsize=y_font_size)
plt.xticks(fontsize=x_font_size)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(root_dir+"./vit_num_params.png", dpi=300)
plt.savefig(root_dir+"./vit_num_params.svg")
plt.close()
