
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
#can only be called once - moved in __init__.py

# try:
#   import pathos
#   pathos_available = True
# except:
#   pathos_available = False

##########################################################################################
#redefine logP to not require args - for (more) convenient mapping
#also function needs to be pickleable and in top level of module for multiprocessing
#post_args = None #set global to None so function is defined
def logP(pars):  return LogPosterior(pars,*post_args)

def DEMC(logPost,gp,args,ch_len,ep=None,N=None,chain_filename='MCMC_chain.npy',burnin=40,n_burnin=5,\
  n_gr=4,gr=1.01,g=None,c=0.01,n_gamma=10,g_glob=0.999,cull=1,dlogP=50,init='norm',Xsamp=None,\
  thin=1,max_ext=0,ext_len=None,parallel=False,n_p=None,var_g=True):
  """
  Differential Evolution MCMC based on Ter Braak (2006).
  
  Essentially an ensemble sampler that uses multiple chains, and bases steps on the other
  chains. For each step, two chains are selected at random (not including the current one)
  and the difference between the two steps are used to generate the direction of the step.
  The amplitude is based on the parameter gamma, which is usually a bit less than 1, but
  can be periodically set to ~1 to enable jumps between modes. A random variable is also
  added to avoid restriction to a hyperplane, and is a small value (c) times the
  covariance matrix. Gamma and c are updated during the burnin phase, and the chain will
  be affine invariant (as steps are based on the currnet distribution) and capable of
  exploring multiple modes assuming the starting distributions are suffiently large.
  
  The main tuning paramters are the number of chains, value for gamma and c. Default is
  that every ten steps gamma is set to ~1. Chains can also be parallised, but this does
  not typically help as the overheads are so poor for multiprocessing which needs to
  copy data for each process.
  
  LogPosterior - log posterior distribution
  gp - array/list of guess parameters
  post_args - additional arguments to posterior, called as LogPoseterior(*post_args)
  ch_len - length of each chain - that is number of generations of DE
  ep - array/list of (initial) steps
  N - number of chains, by default 2 time no of variable dimensions
  chain_filename ['MCMC_chain.npy'] - filename to store results
  burnin - percentage of chain used for burn in - for GR stat and variable parameters
  n_burnin - number of recomputes of gamma and covariance
  n_gr - no of sub chains to use in calculating GR stat
  gr[1.01] - target gr stat
  g[] - acceptance scale, by default depends on no of free params
  c[0.01] - scaling of random step added on to each perturbation to avoid being stuck on hyperplane
  n_gamma[10] - set gamma to g_glob (nearly 1) every n_gamma chains (important for multimodal exploration)
  g_glob [0.99] value of gamma to allow jumps between - should be 1 or nearly 1 (to allow more parameter space to be explored)
  cull[1] - replace very log starting points (>dlogP from max) by repetition
  dlogP[100] - delta logP to cull
  init[norm] - distribution for starting arrays, normal or uniform
    - init can also be a 2D array of samples x n_parameters, to start chains in a specific state
  thin[1] - thin the chains by this amount, ie only record every 'thin' steps of the chain
  max_ext[0] - no of extensions to the chain if gr not met
  ext_len[ch_len/2] - length of extensions  
  parallel[False] - parallelise the chains. This will use multiprocessing to parallelise
    the chain. Generally not always useful as large amount of shared memory to be used, and
    overheads are pretty huge.
    Nonetheless potentially useful and might be a way to speed
    up the mapping functions if I could be bothered.
  n_p - specify the number of separate processes used by multiprocessing
  
  """
  
  #posterior args and LogPosterior must be global to be pickleable (for parallel execution)
  global post_args,LogPosterior
  post_args = args
  LogPosterior = logPost
  
  #might have a solution based on dill without pathos (which seems unreliable)
  #need to dill function and args
  #then run a wrapper function that undills them and runs func as normal, with new p
  #dill the function
# payload = dill.dumps((logPosterior, args))
# #function to undill the payload and run function
# def run_dill_encoded(p):
#     #get the function and common arguments from the dilled 
#     fun, args = dill.loads(payload)
#     return fun(p,*args)
# #run the function
# run_dill_encoded(p_args[0])
# #test if I can pickle the new function
# q = pickle.dumps(run_dill_encoded)

  #remap LogPosterior function for using map
  #note that this will force an unpickleable object
  #and therefore use pathos for multiprocessing
  #could probably use generator objects and zip instead, but benefit from mulitprocessing
  #is limited by copying memory over to each process
#   global post_args2
#   post_args2 = np.copy(post_args)
#  p_args = np.copy(post_args)
#  print p_args
  
  #select map function depending on parallelise behaviour
  if parallel: #to parallelise with multiprocessing needs to pickle
    pool = multiprocessing.Pool(n_p)
    map_func = pool.map
  else:
    map_func = map
  
  #set starting uncertainties
  if ep is None: ep = np.ones(len(gp))/100. #use a small Gaussian ball as default, but no idea of scale
  p,e = np.array(gp),np.array(ep)
  
  #check # of chains
  if N == None:
    N = max((np.array(ep) > 0).sum(),12)
  if N < 8 or (N % n_gr) > 0 or (N % 2) > 0:
    print ("error: N must be at least 8 and divisible by n_gr and 2 [={}]!".format(n_gr))
    return
  
  #set chain len and extension len of chain, default to chain length
  if ext_len == None: ext_len = ch_len // 2
  n_ext = 0
  
  #convert burnin from percentage in chain no
  burnin = int(ch_len * burnin / 100.)
  
  #initialise parameters
  p_acc,L_acc = np.empty((N,p.size)),np.array([-np.inf,]*N)
  p_prop,L_prop = np.empty((N,p.size)),np.empty(N)
  
  #print parameters at start of chain
  PrintParams(N,ch_len,max_ext,ext_len,burnin,LogPosterior,gp,ep)
    
  #arrays for storing results (diluted with a thinning factor)
  ParArr = np.zeros((N,ch_len//thin,p.size))
  PostArr = np.zeros((N,ch_len//thin))
  AccArr = np.zeros((N,ch_len))
  
  #jump distributions, computed in advance - much faster to compute as a block  
  K = np.diag(e**2)
  np.random.seed()
  RandArr = np.random.multivariate_normal(np.zeros(p.size),K,(N,ch_len)) * c
  #set columns to zero after too! - for large K sometimes zero variance parameters have small random scatter
  for n in range(N): RandArr[n,:][:,np.where(np.isclose(e,0.))[0]] = 0.
  RandNoArr = np.random.rand(N,ch_len)

  #set gamma parameter
  if g is None: g = 2.38 / 2. / (e>0).sum()
  gamma = np.ones(ch_len) * g
  if n_gamma > 0: gamma[n_gamma::n_gamma] = g_glob #set every tenth gamma to one to allow jumps between peaks
  gr_str = ""
  
  ####### create draws of other two chains for each step ###############

  #choose two other chains for each n
  #generating random numbers and sorting is the fastest method
  #need to generate extra random nos (and floats), but as quick as other MCMCs
  #np.random.choice is much slower, as can't be run for large arrays simultaneously
  if not parallel: #for normal execution, choose from remainder of chain, ie exclude n
    #get random numbers from 0 to N-2 inclusive
    rc = np.random.random((ch_len,N,N-1)).argsort(axis=2)[:,:,:2]
    #where equal to n, set to N-1 to make uniform distribution from remainder
    rc[np.equal(rc,np.arange(N)[np.newaxis,:,np.newaxis])] = N-1
  else: #for parallel execution, divide into two groups, and simply pick from other group
    rc = np.random.random((ch_len,N,N//2)).argsort(axis=2)[:,:,:2]
    rc[:,:N//2] += N//2
    print("\x1B[3m(running in parallel with {} cores)\x1B[23m".format(multiprocessing.cpu_count() if n_p == None else n_p))
    
  ####### loop over chain generations ###############
  start = time.time()
  
  #initiate chain positions and compute posterior
  if Xsamp is not None: #use samples if given
    assert Xsamp.shape == p_acc.shape, "shape of Xsamp array must match MCMC pars"
    p_acc = Xsamp
  else: 
#    p_acc = init_pars(p,e,N,init)
    p_acc = init_pars(p,e*g,N,init)
  L_acc = np.array(list(map_func(logP,p_acc))) #compute posterior for starting positions
  
  # recompute any in restricted prior space
  cull_list = []
  for n in range(N):
    no_recomp = 0
    while L_acc[n] == -np.inf:
      no_recomp += 1
      if no_recomp > 20:
        cull_list.append(n+1)
        break
      #get new starting point and calculate posterior
      p_acc[n] = init_pars(p,e,type=init)
      L_acc[n] = logP(p_acc[n])
  if len(cull_list) > 0:
    print("warning: after 20 attempts {} chains still initiated in restricted prior space!".format(len(cull_list)))
  
  #cull points lying too far from current max
  if cull: #by replacing with a random choice from the rest
    L_acc_max = L_acc.max()
    #get index for where L_acc is good
    ind_good = np.where((L_acc > L_acc_max-dlogP))[0]
    assert ind_good.size > 1, "must have more than 1 good point for culling (have {})!".format(ind_good.size)
    print("\x1B[3m(culled {}/{} points)\x1B[23m".format(N-ind_good.size,N))
    for n in range(N):
      if L_acc[n] < L_acc_max-dlogP:# or L_acc[n] == -np.inf:
        ind_new = np.random.choice(ind_good)
        L_acc[n] = L_acc[ind_new]
        p_acc[n] = p_acc[ind_new]
         
  #store initial points in chain
  ParArr[:,0],PostArr[:,0] = p_acc,L_acc

  i = 1 #start loop over remaining generations
  PrintBar(i,chain_filename,ch_len,ch_len,AccArr,start)  
  while i<ch_len: #use while loop as ch_len can be adjusted#
    #if i % (ch_len/20) == 0: PrintBar(i,chain_filename,ch_len,ch_len,AccArr,start)  
        
    ################ loop over each individual chain ################

    for n in range(N): #generate proposal step
      p_prop[n] = p_acc[n] + gamma[i] * (p_acc[rc[i][n][0]] - p_acc[rc[i][n][1]]) + RandArr[n][i] #proposal step
#      L_prop[n] = logP(p_prop[n])
    #can parallelise this step using different map functions
    #particular if pots split in 2
    #ie evaluation of multiple posteriors

    L_prop = np.array(list(map_func(logP,p_prop)))
#     for n in range(N): #should be easy to parallelise n here!
#       L_prop[n] = logP(p_prop[n]) #calculate the posterior of at the proposal point
    
    #could introduce a blocked Gibbs sampler here as well? to speed up GP sampling
    
    for n in range(N): #accept the step?
      if RandNoArr[n][i] < np.exp(L_prop[n] - L_acc[n]):
        p_acc[n],L_acc[n] = p_prop[n],L_prop[n]
        AccArr[n][i] = 1 #update acceptance array
      #add new posterior and parameters to chain
      if i%thin==0: ParArr[n][i//thin],PostArr[n][i//thin] = p_acc[n],L_acc[n]
       
    ################ calculate GR stat to monitor progress ################
    #calculate GR statistic after dividing into 2 chains
    if i % (ch_len/20) == 0 or i == ch_len-1:
      if i > burnin:
        #need to trasform the ParArr so independent chains used - bit convoluted but just using previous GR func
        grs = ComputeGRs(e,ParArr[:,burnin//thin:i//thin,:].transpose(1,0,2).reshape(-1,n_gr,p.size).transpose(1,0,2),p.size)
        converged = grs.max() <= gr
        gr_str = " GR[max] = {:.4f}".format(grs.max())
      PrintBar(i,chain_filename,ch_len,ch_len,AccArr,start,extra_st=gr_str)  
    
    ################ Recompute the covariance matrix ################
    #adaptive stepsizes - shouldn't be executed after arrays extended in current form!
    if (i <= burnin) and i % (burnin//n_burnin) == 0:
      lims = [i//thin-burnin//n_burnin,i//thin]
      #recompute K and random arrays
      K = (K + 4.*np.cov(ParArr[:,lims[0]:lims[1]].reshape(-1,p.size),rowvar=0))/5.
      K[np.where(np.isclose(e,0.))],K[:,np.where(np.isclose(e,0.))] = 0.,0. #reset error=0. values to 0.
      RandArr[:,i:] = np.random.multivariate_normal(np.zeros(p.size),K,(N,ch_len-i)) * c
      for n in range(N): RandArr[n,:][:,np.where(np.isclose(e,0.))[0]] = 0.
    
    ################ Update gamma ################
    if var_g and (i <= burnin) and i % (burnin//n_burnin) == 0:
      lims = [i//thin-burnin//n_burnin,i//thin]
      acc = 0.234 #target acceptance
      #rescale gamma for target acceptance
      #introduce min/max limits to avoid changing things too sharply
#      g *= (1./acc) * (AccArr[:,lims[0]:lims[1]].sum() / np.float(lims[1]-lims[0]) / np.float(N))
      g *= (1./acc) * min(0.8,max(0.1,(AccArr[:,lims[0]:lims[1]].sum() / np.float(lims[1]-lims[0]) / np.float(N))))
      gamma[i:] = g
      if n_gamma > 0: gamma[n_gamma::n_gamma] = g_glob #set every tenth gamma to (near) one to allow jumps between peaks
      # print "gam:",gamma[0]
            
    ################ Extend the chains? ################    
    #test convergence and extend chains if needed, unless max extensions has been reached
    if i == ch_len-1:
      
      if converged:
        gr_str = " GR[max] = {:.4f} {}".format(grs.max(),"(converged after {} ext){}".format(n_ext,' '*20) if n_ext>0 else "(converged)")
      
      if not converged:
        #extend chain length + parameter arrays
        if n_ext < max_ext:
          n_ext += 1 #decrement the max extensions int
          gr_str = " GR[max] = {:.4f} (not converged, extending chain x {}...)".format(grs.max(),n_ext)
          ch_len += ext_len  
          #extend arrays to store extra values  
          RandArr = np.concatenate([RandArr,np.random.multivariate_normal(np.zeros(p.size),K,(N,ext_len))],axis=1)
          RandArr[:,i:,np.where(e==0.)[0]] = 0. #set columns to zero after too!
          RandNoArr = np.concatenate([RandNoArr,np.random.rand(N,ext_len)],axis=1)
          AccArr = np.concatenate([AccArr,np.zeros((N,ext_len))],axis=1)   
          ParArr = np.concatenate([ParArr,np.zeros((N,ext_len/thin,p.size))],axis=1)
          PostArr = np.concatenate([PostArr,np.zeros((N,ext_len/thin))],axis=1)
          gamma = np.concatenate([gamma,np.ones(ext_len) * g])
          if n_gamma > 0: gamma[n_gamma::n_gamma] = g_glob #set every tenth gamma to one to allow jumps between peaks
          if not parallel: #create new rc pars in same way as before
            rc_new = np.random.random((ext_len,N,N-1)).argsort(axis=2)[:,:,:2]
            rc_new[np.equal(rc_new,np.arange(N)[np.newaxis,:,np.newaxis])] = N-1
          else: #for parallel execution, divide into two groups, and simply pick from other group
            rc_new = np.random.random((ext_len,N,N//2)).argsort(axis=2)[:,:,:2]
            rc_new[:,:N//2] += N//2
          rc = np.concatenate([rc,rc_new],axis=0)
        else:
          gr_str = " GR[max] = {:.4f} {}".format(grs.max(),"(not converged after {} ext){}".format(n_ext,' '*20) if n_ext>0 else "(not converged)")
      
      PrintBar(i,chain_filename,ch_len,ch_len,AccArr,start,extra_st=gr_str)  
      
      ####### end loop over each of N chains ###########
    
    #increment i
    i += 1
    
    ####### end loop over all chains ###########

  PrintBar(i,chain_filename,ch_len,ch_len,AccArr,start,True,"",end_st="Final chain len = {}; Samples = {}".format(i,i*N))
  
  #save the chains in a single file
  #save the chain - in chunks of 'generations' - ie N chains repeated evolving in time
  #a single chain n will be stored in X[n::N,:]
  np.save(chain_filename,np.concatenate([PostArr.T.flatten().reshape(-1,1),ParArr.transpose((1,0,2)).reshape(-1,p.size)],axis=1))
  
  #check acceptance ratios for all chains
  acc_arr = AccArr[:,burnin:].sum(axis=1) / np.float(ch_len-burnin)
  print (u"\x1B[3m final acceptance ratio: {:.2f}% (min: {:.2f}%)\x1B[23m".format(acc_arr.mean()*100.,acc_arr.min()*100.))
  
  ####### end loop over chains ############
  print('-' * 80)
  
  #return expectation values and uncertainties
  mean,stdev = AnalyseChainsDEMC(PostArr.flatten(),ParArr[:,burnin//thin:i//thin,:].transpose(1,0,2).reshape(-1,n_gr,p.size).transpose(1,0,2),verbose=True)
  
  #close the pool for parallel execution to avoid zombie processes
  if parallel: pool.close()
  
  #reset parameters where e=0 to avoid numerical errors
  mean[np.where(np.isclose(e,0.))] = p[np.where(np.isclose(e,0.))]
  stdev[np.where(np.isclose(e,0.))] = 0.
  
  #and return
  return mean,stdev
  
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

def PrintBar(i,ch_file,ch_len,var_ch_len,AccArr,start,finish=False,extra_st="",end_st=""):
  ts = time.time()-start
  print("Running DEMC:" + extra_st)
#  a_str = "" if i <= ch_len/5 else ", acc = %.2f%%  " % (100.*np.float(AccArr.flatten()[ch_len/5:i].sum())/(i-ch_len/5.))
  acc_flattened = AccArr[:,ch_len//5:i].flatten()
  a_str = "" if i <= ch_len//5 else ", acc = %.2f%%  " % (100.*np.float(acc_flattened.sum())/len(acc_flattened))
  print(u" chain: '%s' \033[31m%-21s\033[0m t = %dm %.2fs%s" % (ch_file,'#'*(i//(var_ch_len//20)+1),ts // 60., ts % 60.,a_str))
  sys.stdout.write('\033[{}A'.format(2))
  if finish: print("\n"*2 + end_st)

##########################################################################################

def PrintParams(N,ch_len,max_ext,ext_len,burnin,LogPosterior,p,e):

  print('-' * 80)
  print("DE-MC chain runnning...")
  print(" No Chains: {}".format(N))
  print(" Chain Length: {} {}".format(ch_len,"(ext: {}x{})".format(max_ext,ext_len) if max_ext>0 else ""))
  print(" Burn in: {}".format(burnin))
  print(" Samples: {}".format(ch_len*N))
  #print " Max {} extensions of length {}".format(max_ext,ext_len)
  #if(adapt_limits[2]): print " Relative-step adaption limits: (%d,%d,%d)" % (adapt_limits[0],adapt_limits[1],adapt_limits[2])
  #if(glob_limits[2]): print " Global-step adaption limits: (%d,%d,%d)" % (glob_limits[0],glob_limits[1],glob_limits[2])
  #print " Computing {} chains simultaneously: {}".format(len(ch_filenames),ch_filenames)
  print(" Posterior probability function: {}".format(LogPosterior))
  print(" Function params <value prop_size>:")
  for q in range(len(p)):
    print("  p[{}] = {} +- {}".format(q,p[q],e[q]))
  print('-' * 80)

##########################################################################################

def ComputeGRs(ep,ParArr,conv=0):
  """
  Compute the Gelman and Rubin statistic and errors for all variable parameters
  Assumes chains x time x pars
  
  """
  
  p = np.where(np.array(ep)>0)[0]
  GRs = np.zeros(len(p))
  
  for q,i in enumerate(p):
    
    #get mean and variance for the two chains
    mean = ParArr[:,:,i].mean(axis=1)
    var = ParArr[:,:,i].var(axis=1)
    
    #get length of chain
    L = len(ParArr[0,:,0])
    
    #and calculate the GR stat
    W = var.mean(dtype=np.float64) #mean of the variances
    B = mean.var(dtype=np.float64) #variance of the means
    GR = np.sqrt((((L-1.)/L)*W + B) / W) #GR stat
    
    GRs[q] = GR
  
  return GRs

##########################################################################################

def AnalyseDEMC(file,burnin=40,n_gr=4,verbose=True):
  """
  Wrapper to call AnalyseChainsDEMC given a filename.
  
  """
  if file[-4:] == '.npy':
    X = np.load(file)
  else:
    X = np.load(file+'.npy')

  #get chain length and n_pars from 2D array
  ch_len = X.shape[0]
  n_par = X.shape[1] - 1
  burn = int(ch_len * 40 / 100.) #get burnin to filter out
  
  #reshape X into chains x ch_len pars
  Pars = X[burn:,1:].reshape(-1,n_gr,n_par).transpose(1,0,2)
  L = X[:burn,0] #get flattened likelihood array
  
  return AnalyseChainsDEMC(L,Pars,verbose=verbose)


def AnalyseChainsDEMC(L,X,verbose=False):
  """
  Simple analysis function, assuming X contains 4chains x time x [logP + par]
    
  """
    
  #set empty arrays for the mean and gaussian errors (for returning)
  no_pars = X.shape[-1]
  mean = np.empty(no_pars)
  median = np.empty(no_pars)
  gauss_err = np.empty(no_pars)
  GR = np.zeros(no_pars)
  pos_err = np.empty(no_pars)
  neg_err = np.empty(no_pars)

  print ("MCMC Marginalised distributions:")
  print (" par = mean gauss_err [med +err -err]: GR")
  for i in range(no_pars): #loop over parameters and get parameters, errors, and GR statistic
    mean[i],gauss_err[i] = X[...,i].mean(),X[...,i].std()
    sorted_data = np.sort(X[...,i].flatten())
    median[i] = np.median(X[...,i])
    pos_err[i] = sorted_data[int((1.-0.159)*sorted_data.size)]-median[i]
    neg_err[i] = median[i]-sorted_data[int(0.159*sorted_data.size)]
  GR[gauss_err>0] = ComputeGRs(gauss_err,X)
  
  if verbose:
    for i in range(no_pars):
      print (" p[%d] = %.7f +- %.7f [%.7f +%.7f -%.7f]: GR = %.4f" % (i,mean[i],gauss_err[i],median[i],pos_err[i],neg_err[i],GR[i]))
  
  #calculate evidence approximations
  m = X.reshape(-1,X.shape[-1]).mean(axis=0)
  K = np.cov(X.reshape(-1,X.shape[-1])-m,rowvar=0) #better to mean subtract first to avoid rounding errors
  #get max likelihood
  logP_max = np.nanmax(L)
  
  #first must compress the covariance matrix as some parameters are fixed!
  var_par = np.diag(K)>0
  Ks = K.compress(var_par,axis=0);Ks = Ks.compress(var_par,axis=1)
  D = np.diag(Ks).size #no of dimensions
  sign,logdetK = np.linalg.slogdet( 2*np.pi*Ks ) # get log determinant
  logE = logP_max + 0.5 * logdetK #get evidence approximation based on Gaussian assumption
  print ("Gaussian Evidence approx:")
  print (" log ML =", logP_max)
  print (" log E =", logE)
  logE_BIC = logP_max
  print (" log E (BIC) = log ML - D/2.*np.log(N) =", logP_max, "- {}/2.*np.log(N)".format(D))
  logE_AIC = logP_max - D
  print (" log E (AIC) = log ML - D =", logE_AIC, "(D = {})".format(D))
    
  return mean,gauss_err

##########################################################################################
