#this is the dense part, for energy fitting
#no normalization
#classical Hamiltonian
#the growth rate of layer number is controlled
#heisemnberg
#############################
use the neural network structure in hs_densenet_structure.py
1. python hs_densenet_train.py L r  num_layers growth_rate
2. change num_layers in plt_log.py, then
    python plt_log.py
3. change num_layers in growth_rate_densenet_no_bn_test.py, then
    python _test.py
4. change num_layers in extract_test.py, then
   python extract_test.py
5. plot loss:
    python plt_std_loss.py