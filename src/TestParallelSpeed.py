
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time
import multiprocessing
#this line required for python3.8+, due to change in default method
#need to double check it doesn't break python 2 version
#and there are I think more up to date methods to multiprocess with py3.8
#multiprocessing.set_start_method("fork")
#moved to init file, can only be called once
#also needed to change list(map) as map now returns an iterator

def logP(pars):
  return LogPosterior(pars,*post_args)

def DEMCSpeedTest(logPost,gp,args,ep,N,parallel=False,n_p=4,init='norm'):
  """
  Simple function that uses the same parallelisation method as DEMC to run functions
   and test speed.
  
  """
#  print('running...')
  
  #posterior args and LogPosterior must be global to be pickleable (for parallel execution)
  global post_args,LogPosterior
  post_args = args
  LogPosterior = logPost
    
  #select map function depending on parallelise behaviour
  if parallel: #to parallelise with multiprocessing needs to pickle
    pool = multiprocessing.Pool(n_p)
    map_func = pool.map
  else:
    map_func = map
  
  p,e = np.array(gp),np.array(ep)    
  p_acc = init_pars(p,e,N,init)
  L_acc = np.array(list(map_func(logP,p_acc))) #compute posterior
  
  if parallel: pool.close()

  return L_acc
    
##########################################################################################

def init_pars(p,e,n=1,type='norm'):
  """
  return samples in n x n_par array depending on initialisation type
  
  """
  
  if type == 'norm':
    samples = np.random.multivariate_normal(p,np.diag(e**2),n)
  if type == 'uniform':
    #samples = np.array([p + np.random.uniform(-0.5,0.5,p.size) * e for i in range(n)]).squeeze()
    samples = np.array([p + np.random.uniform(-0.5,0.5,p.size) * e for i in range(n)])
  
  #set samples to be fixed where error is zero, to avoid numerical errors
  samples[:,np.where(np.isclose(e,0.))[0]] = p[np.where(np.isclose(e,0.))[0]]
  
  return samples

##########################################################################################

