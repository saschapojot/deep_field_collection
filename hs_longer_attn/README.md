#this is the efnn part, for energy fitting
#no normalization
#classical Hamiltonian
#heisenberg
#############################
use the neural network structure in hs_longer_attn_structure.py
1. python hs_longer_attn_train.py L r  num_layers embed_dim num_heads
2. change num_layers, embed_dim in plt_log.py, then
    python plt_log.py
3. change num_layers, epoch_num, in hs_longer_attn_test.py, then
    python hs_longer_attn_test.py
4. change num_layers, epoch_num,  in extract_test.py, then
   python extract_test.py
5. plot loss: change epoch_num in plt_std_loss.py, 
    python plt_std_loss.py