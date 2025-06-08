#this is the dense part, for energy fitting
#no normalization
#classical Hamiltonian
#the growth rate of layer number is controlled
#heisemnberg
#############################
use the neural network structure in hs_efnn_structure.py
1. python hs_densenet_train.py L r  num_layers num_neurons
2. change num_layers, num_neurons in plt_log.py, then
    python plt_log.py
3. change num_layers, epoch_num, in hs_efnn_test.py, then
    python hs_efnn_test.py
4. change num_layers, epoch_num,  in extract_test.py, then
   python extract_test.py
5. plot loss: change epoch_num in plt_std_loss.py, 
    python plt_std_loss.py