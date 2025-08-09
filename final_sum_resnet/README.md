#this is the resnet part, for energy fitting
#no batch normalization
#classical Hamiltonian
#final layer is sum

network definition is in structure.py
1. training: 
    python train.py L r num_layers num_neurons
2. change num_layers in plt_log.py, then
    python plt_log.py
3. performance on test set: 
    in test.py, change num_layers, then
    python test.py
4. change num_layers, epoch_num,  in extract_test.py, then
   python extract_test.py
5. plot loss: change epoch_num in plt_std_loss.py, 
    python plt_std_loss.py