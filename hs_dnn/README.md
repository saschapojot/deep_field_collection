#this is the dnn part, for energy fitting
#no normalization
#classical Hamiltonian
#heisenberg

#############################
use the neural network structure in hs_dnn_structure.py

1. python hs_dnn_train.py L r  num_layers num_neurons
2. change num_layers, num_neurons in plt_log.py, then
    python plt_log.py
3. change num_layers, epoch_num, in dnn_test.py, then
    python hs_dnn_test.py