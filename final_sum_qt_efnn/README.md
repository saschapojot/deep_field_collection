#this is efnn, quantum Hamiltonian, pbc, no batch normalization

#the neural network structure is defined in structure.py

1. training: 
    python train.py num_epochs C step_num_after_S1
2. compute test results for different epochs:
    in test_over_epochs.py, change N, C, layer
    python test_over_epochs.py
3. plot test results for  1 epoch:
    in plt_one_epoch.py, change  N, C, layer
    python plt_one_epoch.py
4. run test on larger lattice in 1 run, set C:
    python all_N_all_layers_large_lattice_test.py
5. python plt_all_layers_larger_lattice_test_performance.py