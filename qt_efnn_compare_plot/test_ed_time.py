import numpy as np
import sys
from datetime import datetime

#this script tests the computation time by ED
#model is pbc


def generate_uniform_matrix(N, low=0, high=2*np.pi, seed=None):
    """
    Generates an N x N matrix with elements uniformly distributed between low and high.

    Parameters:
    - N (int): Size of the matrix (number of rows and columns).
    - low (float): Lower bound of the uniform distribution (default is 0).
    - high (float): Upper bound of the uniform distribution (default is 2π).
    - seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - numpy.ndarray: An N x N matrix with uniformly distributed elements.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, size=(N, N))





def generate_Sigmax_Sigmay_Sigmaz(N,seed=None):
    """

    :param N: length of lattice
    :param seed:
    :return: components of classical spins over 2d lattice, 3 matrices are x, y , z components
    """
    Theta_mat=generate_uniform_matrix(N,0,np.pi,seed)
    Phi_mat=generate_uniform_matrix(N,0,2*np.pi,seed)

    sin_Theta_mat=np.sin(Theta_mat)

    sin_Phi_mat=np.sin(Phi_mat)

    cos_Theta_mat=np.cos(Theta_mat)

    cos_Phi_mat=np.cos(Phi_mat)

    Sigma_x=sin_Theta_mat*cos_Phi_mat
    Sigma_y=sin_Theta_mat*sin_Phi_mat

    Sigma_z=cos_Theta_mat

    Sigma_combined = np.stack([Sigma_x, Sigma_y, Sigma_z])

    return Sigma_combined


argErrCode=3
if (len(sys.argv)!=2):
    print("wrong number of arguments")
    print("example: python genData_qt.py N")
    exit(argErrCode)


N=int(sys.argv[1])
#construct T0, T1 mat
T0_mat=np.zeros((N**2,N**2),dtype=float)

for r in range(0,N):
    for s in range(0,N):
        r_this=r
        r_next=(r+1)%N

        rowInd=r_this*N+s
        colInd=r_next*N+s

        T0_mat[rowInd,colInd]=1

        T0_mat[colInd,rowInd]=1
T1_mat=T0_mat
#construct T2, T3 mat

T2_mat=np.zeros((N**2,N**2),dtype=float)
for r in range(0,N):
    for s in range(0,N):
        s_this=s
        s_next=(s+1)%N

        rowInd=r*N+s_this

        colInd=r*N+s_next

        T2_mat[rowInd,colInd]=1

        T2_mat[colInd,rowInd]=1

T3_mat=T2_mat


def gen_Gamma_mat(T0_mat,T1_mat,T2_mat,T3_mat,Sigma_combined,t,J,mu,I):
    """

    :param T0_mat:
    :param T1_mat:
    :param T2_mat:
    :param T3_mat:
    :param Sigma_combined:
    :param t:
    :param J:
    :param mu:
    :param I:
    :return: Gamma(S)
    """
    # print(f"Sigma_combined.shape={Sigma_combined.shape}")
    Sigma_x=Sigma_combined[0,:,:]#N by N matrix

    Sigma_y=Sigma_combined[1,:,:]# N by N matrix

    Sigma_z=Sigma_combined[2,:,:]# N by N matrix

    Sx_vec=Sigma_x.flatten()# length N**2 vector

    Sy_vec=Sigma_y.flatten()# length N**2 vector

    Sz_vec=Sigma_z.flatten()# length N**2 vector

    Sx=np.diag(Sx_vec,k=0)# N**2 by N**2 matrix

    Sy=np.diag(Sy_vec,k=0)# N**2 by N**2 matrix

    Sz=np.diag(Sz_vec,k=0)# N**2 by N**2 matrix




    block00=-t*T0_mat-t*T2_mat-1/2*J*Sz-mu*I# N**2 by N**2 matrix

    block01=-1/2*J*Sx+1j*1/2*J*Sy# N**2 by N**2 matrix

    block10=-1/2*J*Sx-1j*1/2*J*Sy# N**2 by N**2 matrix

    block11=-t*T1_mat -t*T3_mat+1/2*J*Sz-mu*I# N**2 by N**2 matrix

    Gamma_mat=np.block(
        [[block00,block01],
        [block10,block11]]
    )# 2*N**2 by 2*N**2 matrix

    return Gamma_mat


def vec_2_E(eig_vals,T):
    """

    :param eig_vals:
    :param T:
    :return:
    """
    # print(f"eig_vals={eig_vals}")
    exp_vals=np.exp(-1/T*eig_vals)

    E=-T*np.sum(np.log(1+exp_vals))
    return E

def assemble_input_data(N,T0_mat,T1_mat,T2_mat,T3_mat,t,J,mu,I_N2,seed=None):
    Sigma_combined_tmp = generate_Sigmax_Sigmay_Sigmaz(N, seed)
    gm_mat_tmp = gen_Gamma_mat(T0_mat, T1_mat, T2_mat, T3_mat, Sigma_combined_tmp, t, J, mu, I_N2)
    return gm_mat_tmp



def gen_one_data(Gamma_matrix,T):
    eigValsTmp, eigVecsTmp = np.linalg.eigh(Gamma_matrix)
    E_tmp = vec_2_E(eigValsTmp, T)
    return E_tmp
t=1

J=16*t
mu=-8.3*t
T=0.1*t
I_N2=np.eye(N**2)
# gm_mat_tmp=assemble_input_data(N,T0_mat,T1_mat,T2_mat,T3_mat,t,J,mu,I_N2)
# tStart=datetime.now()
# E_tmp=gen_one_data(gm_mat_tmp,T)
# tEnd=datetime.now()
#
# print(f"N={N}, generate E_MC time: {tEnd-tStart}")

num_samples = 17
total_duration = 0

for i in range(num_samples):
    # 1. Generate Matrix (New random configuration)
    gm_mat_tmp = assemble_input_data(N, T0_mat, T1_mat, T2_mat, T3_mat, t, J, mu, I_N2)
    # 2. Compute Energy (ED)
    loop_start = datetime.now()
    E_tmp = gen_one_data(gm_mat_tmp, T)
    # End Timer
    loop_end = datetime.now()
    duration = (loop_end - loop_start).total_seconds()
    total_duration += duration
    print(f"Sample {i + 1}: E = {E_tmp:.4f} | Time: {duration:.4e} seconds")

print("-" * 40)
print(f"Average Time per Sample: {total_duration / num_samples:.4e} seconds")