#this is densenet, quantum Hamiltonian, pbc, with batch normalization

#the neural network structure is defined in qt_densenet_structure.py

1. training: 
    python qt_densenet_train.py num_epochs C step_num_after_S1
2. compute test results for different epochs:
    in qt_resnet_test_over_epochs.py, change N, C, layer
    python qt_resnet_test_over_epochs.py
3. plot test results for  1 epoch:
    in plt_one_epoch.py, change  N, C, layer
    python plt_one_epoch.py
4. run test on larger lattice in 1 run:
    python qt_resnet_all_N_all_layers_large_lattice_test.py