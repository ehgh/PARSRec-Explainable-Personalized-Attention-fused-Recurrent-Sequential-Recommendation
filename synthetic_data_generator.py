# Copyright (c) Ehsan G., Mohammad M., Ashwin A.
#
#        _____     _____     _____     _____    _____     _____    _____                                           
#       |  _  |   |  _  |   |  _  |   |  ___   |  _  |   |  ___   |  ___
#       | |_| |   | |_| |   | |_| |   | |___   | |_| |   | |__    | |                                             
#       |   __|   |  _  |   |  _  /   |___  |  |  _  /   | |___   | |___                                                      
#       |  |      | | | |   | | \ \    ___| |  | | \ \   |_____   |_____                                           
# 
# This source code is licensed under the MIT license found in the
#
# Description: an implementation of synthetic session data generation
# 
# References:
# Ehsan Gholami, Mohammad Motamedi, and Ashwin Aravindakshan. 2022.
# PARSRec: Explainable Personalized Attention-fused Recurrent Sequential
# Recommendation Using Session Partial Actions. In Proceedings of the 28th
# ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD
# ’22), August 14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA,
# 11 pages. https://doi.org/10.1145/3534678.3539432


import os
import random
import swifter
import argparse
import numpy as np
from tqdm import tqdm
from random import shuffle
from utils import load_dataset
from scipy.linalg import block_diag
from scipy.stats import weibull_min
from timeit import default_timer as timer

tqdm.pandas()

np.set_printoptions(precision = 2, suppress = True, linewidth=500)

#generate inter-category correlation matrix 
omegas = []

D1 = [[1, -0.25, -0.25, -0.25],
      [-0.25, 1, -0.25, -0.25],
      [-0.25, -0.25, 1, -0.25],
      [-0.25, -0.25, -0.25, 1]] 
D2 = [[1, 0.2, 0.5, 0.8],
      [0.2, 1, 0.4, 0.7],
      [0.5, 0.4, 1, 0.6],
      [0.8, 0.7, 0.6, 1]]
D3 = [[1, -0.4, 0.5],
      [-0.4, 1, 0.5],
      [0.5, 0.5, 1]]
D4 = [[1, 0.5, -0.5],
      [0.5, 1, -1],
      [-0.5, -1, 1]]
D5 = [[1, 0.5, -0.3],
      [0.5, 1, 0.5],
      [-0.3, 0.5, 1]]
D6 = [[1]]
D7 = [[1, 0.99],
      [0.99, 1]]
mat = (block_diag(D1, D2, D3, D4, D5, D6, D7),)
omegas.append(block_diag(*mat))

D1 = [[1, -0.25, -0.25, -0.25],
      [-0.25, 1, -0.25, -0.25],
      [-0.25, -0.25, 1, -0.25],
      [-0.25, -0.25, -0.25, 1]] 
D2 = [[1, 0.2, 0.5, 0.8],
      [0.2, 1, 0.4, 0.7],
      [0.5, 0.4, 1, 0.6],
      [0.8, 0.7, 0.6, 1]]
D3 = [[1, -0.4, -0.5],
      [-0.4, 1, -0.5],
      [-0.5, -0.5, 1]]
D4 = [[1, 0.5, -0.5],
      [0.5, 1, -1],
      [-0.5, -1, 1]]
D5 = [[1, 0.5, 1.0],
      [0.5, 1, 0.5],
      [1.0, 0.5, 1]]
D6 = [[1]]
D7 = [[1, -0.5],
      [-0.5, 1]]
mat = (block_diag(D1, D2, D3, D4, D5, D6, D7),)
omegas.append(block_diag(*mat))

def sigma_per_cat_vine(eta=0, shape=None, tau=None):
  '''
  generates intra-category correlation matrix
  Sigma_c = (tau * I_c) * omega_c * (tau * I_c)
  omega_c ~ Beta(0.2, 1) (symmetric with diagonal 1)
  '''
  if shape is None:
    shape = (args.C, args.Jc, args.Jc)
  Sigma_c = np.empty(shape)
  sz = shape[-1]
  if not tau:
    tau = args.tau0
  I_c = np.identity(sz)

  for category in range(shape[0]):
    
    beta = eta + (sz-1)/2
    #storing partial correlations
    P = np.zeros((sz,sz))
    #correlation matrix
    omega_c = np.eye(sz)

    for k in range(0, sz-1):
      beta = beta - 1/2
      for i in range(k+1, sz):
        #partial correlations from beta distribution
        P[k,i] = np.random.beta(0.2, 1)
        p = P[k,i]
        #converting partial correlation to raw correlation
        for l in range(k-1,-1,-1):
            p = p * np.sqrt((1-P[l,i]**2)*(1-P[l,k]**2)) + P[l,i]*P[l,k]
        omega_c[k,i] = p
        omega_c[i,k] = p
    
    tau_I_c = tau * I_c
    sigma_c = np.matmul(tau_I_c, omega_c)
    sigma_c = np.matmul(sigma_c, tau_I_c)
    Sigma_c[category, :, :] = sigma_c
  return Sigma_c

def generate_utility():
  '''
  products are selected by MNP model per category
  u_ijt = alpha_ij - beta * p_j + e_ijt
  alpha_ij ~ MVN(0, Sigma_c)
  Sigma_c = (tau * I_c) * omega_c * (tau * I_c)
  e_ijt ~ N(0, sigma0)
  '''
  #correlation matrix
  Sigma_c = sigma_per_cat_vine()
  
  #base utility per person-time-product
  alpha_c_itj = np.empty((args.C, args.I, args.T, args.Jc))
  for category in range(args.C):
    alpha_c_ij = np.random.multivariate_normal(
      mean = [0] * args.Jc, cov = Sigma_c[category, :, :], size = (args.I))
    alpha_c_itj[category, :, :, :] = np.repeat(alpha_c_ij[:, None, :], args.T, axis=1)

  #transpose time and product axis for proper order later
  alpha_c_ijt = alpha_c_itj.transpose(0,1,3,2)

  #random error term
  e_c_ijt = np.random.normal(0, args.Sigma0, size = (args.C, args.I, args.Jc, 
                             args.T))
  
  #price disutility
  p_c = np.random.lognormal(mean=0.5, sigma=0.1, size=args.C)
  p_c_j = np.random.uniform(low=p_c/2 , high=2*p_c, size=(args.Jc, args.C)).T
  
  #utility function
  utility_c_ijt = alpha_c_ijt - args.Beta * p_c_j[:,None,:,None] + e_c_ijt
  
  return utility_c_ijt, p_c_j

def select_categories():
  '''
  purchasing categories is selected by MVN(Gamma0, Omega)
  output is in shape of (I * T * C):
     I: number of consumers
     T: number of times
     C: number of products per category
  '''
  z = []

  #average and covariance matrices
  for omega in omegas:
    omega = omega[:args.C, :args.C]
    means = [args.Gamma0] * args.C
    assert np.all(np.linalg.eigvals(omega) > -1e-15), 'eigen values must be all positive (semi-positive matrix). Eigen values:\n{}'.format(np.linalg.eigvals(omega))
    assert omega.shape[0] == omega.shape[1] and omega.shape[0] >= args.C, 'category covariance matrix must be square matrix and dimension at least the number of categories'
    z_ = np.random.multivariate_normal(mean = means, 
                                      cov = omega, size = (1*args.I//2, args.T),
                                      check_valid='warn', tol=1e-8)
    z.append(z_)
  
  return np.concatenate(z, axis=0)

def data_generator():
  '''
  generate sessions in two steps:
  1 - choosing purchasing categories
  2 - selecting products from each category
  '''

  #create directory for output files
  if not os.path.isdir(args.data_directory):
    os.makedirs(args.data_directory)
    
  #file to save data
  synthetic_file = open(os.path.join(args.data_directory, 'synthetic.txt'), 'w')
  synthetic_file.write('user_id,time,session_length,session_items\n')
  start = timer()
  
  ##### select purchasing categries #####
  y = select_categories()
  print('catagories selected' + '.'*50)
  
  #break down user generation due to memory limit
  assert (args.I%args.batch_size==0), 'number of users, I, must be dividable by batch_size'
  args.I //= args.batch_size
  
  for batch in range(args.batch_size):
    print('user partition {}'.format(batch))

    ##### session sizes #####
    #generate list of session sizes from a fitted weibull distribution
    sessions_sz = np.ceil(weibull_min.rvs(0.8046973517087279, 
                                          loc=2, 
                                          scale=1.4738173215687265, 
                                          size=args.I*args.T)).astype(int)
    #filter out small and too large sessions
    assert args.max_session_len <= args.C, 'number of product categories cannot be less than session size'
    #sessions_sz = np.random.randint(args.min_session_length, args.max_session_length, size=args.I*args.T)
    sessions_sz[sessions_sz<args.min_session_len] = args.min_session_len
    sessions_sz[sessions_sz>args.max_session_len] = args.max_session_len
    
    ###### select products per purchasing categories #####
    utility_c_ijt, p_c_j = generate_utility()
    print('utilities generated')

    ##### generate sessions for customer i at time t #####
    for t in range(args.T):

      for i in range(args.I):

        session = []
        
        #pick top categories upto session size
        session_sz = sessions_sz[t * args.I + i]
        #pick the product with highest utility in chosen categories
        for category in np.argpartition(-y[batch * args.I + i, t, :], session_sz)[:session_sz]:
            idx = np.argmax(utility_c_ijt[category, i, :, t])
            j = category * args.Jc + idx + 2 #+2 to account for SOB(0) and EOB(1)
            session.append(j)

        #shuffle items of session randomly
        random.shuffle(session)

        #save to file
        synthetic_file.write(','.join(map(str, [batch*args.I+i, t, len(session)] + session)) + '\n')
          
  print('data generation time:{}'.format(timer() - start))
  synthetic_file.close()
  
def main(**kwargs):

  parser = argparse.ArgumentParser()
  parser.add_argument("-I", type = int, help = "Number of consumers", 
                      default = 100)
  parser.add_argument("-T", type = int, help = "Number of sessions per consumer (time)", 
                      default = 50)
  parser.add_argument("-C", type = int, help = "Number of categories",
                      default = 20)
  parser.add_argument("-Jc", type = int, 
                      help = "Number of products per category",
                      default = 15)
  parser.add_argument("-Gamma0", type = float, 
                      help = "Category purchase incidence base utility",
                      default = -0.5)
  parser.add_argument("-Sigma0", type = float, 
                      help = "Standard deviation for error term in MNP",
                      default = 1)
  parser.add_argument("-tau0", type = float, 
                      help = "Standard deviation for covariance matrices in MNP",
                      default = 2)
  parser.add_argument("-Beta", type = float, 
                      help = "Price sensitivity",
                      default = 2)
  parser.add_argument("-min-session-len", type = int, 
                      help = "minimum session size",
                      default = 3)
  parser.add_argument("-max-session-len", type = int, 
                      help = "maximum session size",
                      default = 11)
  parser.add_argument("-batch-size", type = int, 
                      help = "consumer minibatch size (for parallel processing)",
                      default = 1)
  parser.add_argument("-hist-length", type = int, 
                      help = "number of previous sessions to include in the user's historical actions",
                      default = 1)
  parser.add_argument("-numpy-rand-seed", help = "random seed for reproducibility", type=int, default=123)
  parser.add_argument("-cpu-count", type=int, 
                      help = "number of cpu cores to use in parallel processing (used for extracting user history)",
                      default=1)
  parser.add_argument("-data-directory", type = str, 
                      help = "directory to store the data",
                      default = 'data')

  global args
  args = parser.parse_args()
  args_dict = vars(args)
  for k, v in kwargs.items():
    args_dict[k] = v

  np.random.seed(args.numpy_rand_seed)
  data_generator()

  #convert data into binary data files
  print('converting dataset to binary files ' + '.' * 50)
  load_dataset(os.path.join(args.data_directory, 'synthetic.txt'), 
              items_dtype=np.int16, history_len=args.hist_length, 
              dataset='synthetic', cpu_count=args.cpu_count,
              data_directory=args.data_directory)

if __name__ == "__main__":
  main()
