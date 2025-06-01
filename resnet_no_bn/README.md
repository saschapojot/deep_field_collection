#this is the resnet part, for energy fitting
#no batch normalization

##################################### 
For final q=sum, use neural network structure in no_bn_resnet_structure_final_sum.py
1.  training:
    python no_bn_resnet_final_sum_train.py L r num_layers num_neurons
2. plot training log:
    in plt_log.py, change num_layers, then
    python plt_log.py
3. performance on test set: 
    in no_bn_resnet_final_sum_test.py, change num_layers, then
    python no_bn_resnet_final_sum_test.py
4. extract test set performance results for 1 num_layers, all num_neurons:
    in extract_test.py, change num_layers, then
    python extract_test.py
5. plot loss:
    python plt_std_loss.py