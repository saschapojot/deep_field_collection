#this is the resnet part, for energy fitting

##################################### 
For final q=sum, use neural network structure in resnet_structure_final_sum.py
1. python resnet_final_sum_train.py L r num_layers num_neurons
2. plot training log:
    in plt_log.py, change num_layers, then
    python plt_log.py
3. performance on test set: 
    in resnet_final_sum_test.py, change num_layers, then
    python resnet_final_sum_test.py
4. extract test set performance results for 1 num_layers, all num_neurons:
   in extract_test.py, change num_layers, then
    python extract_test.py
5. plot loss:
    python plt_std_loss.py