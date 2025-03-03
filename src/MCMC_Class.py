
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time
import multiprocessing
import sys
from tqdm.auto import tqdm
try:
  import dill
  dill_available = True
except ImportError: dill_available = False

if sys.version_info >= (3, 8):
  #this basically creates a copy of the multiprocessing API with fork context
  #means I don't conflict with the existing multiprocessing method called
  #resetting the method via set_start_method can only be done once!
  multiprocessing = multiprocessing.get_context("fork")
#this line required for python3.8+, due to change in default method
#need to double check it doesn't break python 2 version
#and there are I think more up to date methods to multiprocess with py3.8
#multiprocessing.set_start_method("fork")
#can only be called once - moved in __init__.py


##########################################################################################
#redefine logP to not require args - for (more) convenient mapping
#also function needs to be pickleable and in top level of module for multiprocessing
#post_args = None #set global to None so function is defined
def logP(pars):  return LogPosterior(pars,*post_args)

class mcmc(object):
  """
  Creates OO version of MCMCs. Idea is that I can hold on to all of the parameters and
  save the whole state, and incrementally improve chains etc.
  Will also (eventually) extend to other modes of MCMC
  
  """
  
  #set defaults - all of which can be overwritten in __init__ or initialise
  #this setup just makes it easier to change in different methods
  #should not include pars that requires anything more complex than being directly set
  defaults = dict(var_K=True,target_acc=0.234,target_gr=1.01,dlogP=50,cull_attempts=10,
                  n_p=multiprocessing.cpu_count()//2,
                  n_extensions=2,
                  orth=False,global_K=True, #MH defaults
                  var_g=True,g=None,c=1e-5,n_gamma=10,g_glob=0.999, #DEMC defaults
                  gibbs_ind = None, #BlockedGibbs defaults
                  )
    
  def __init__(self,logPost,args=[],N=2,mode='DEMC',filename=None,n_burnin=5,\
    n_gr=None,cull=None,thin=1,parallel=False,n_p=None,**kwargs):
    """
    Initialise the MCMC
    
    """
  
    #posterior args and LogPosterior must be global to be pickleable (for parallel execution)
    global post_args,LogPosterior
    post_args = args
    LogPosterior = logPost

    #overwrite defaults if any provided
    for key,value in self.defaults.items():
      setattr(self,key,kwargs.get(key,value))
    
    #define map function depending on parallelise behaviour
    if parallel: #to parallelise with multiprocessing needs to pickle
      self.parallel = True
      #if n_p is None: n_p = multiprocessing.cpu_count()//2 # use number of physical cores by default
      #self.n_p = n_p
      # pool and map_func are assigned dynamically for parallel execution
    else:
      self.parallel = False
      self.map_func = map
    
    #setup basic chain parameters
    self.N = N # no of walkers/chains
    self.mode = mode
    self.cull = cull
    self.set_mode(self.mode) #set type of MCMC + check a few things
    self.thin = thin #parameter to thin the chains - ie store every 'thin' iterations
    #culling parameters
    
    #burnin parameters
    #self.var_K = var_K # vary K in burnin
    self.n_burnin = n_burnin # default number of times to update parameters during burnin
    
    #gr parameters for automatic checking
    self.n_gr = n_gr #number of chains to calculate GR stat
    
    #save parameters after each run of chain if filename is set
    self.filename = filename
    
    #pass arguments to setup if burn, chain, or extend is included
    #allow for full execution in one line
    setup_kwds = dict(X=None,p=None,e=None,K=None,dist='norm',burn=0,chain=0,extend=0)
    for key,value in setup_kwds.items():
      if key in kwargs: setup_kwds[key] = kwargs[key]
    #print(setup_kwds)
    #run setup as well if certain keywords combos are added
    if setup_kwds['burn']>0 or setup_kwds['chain']>0 or setup_kwds['X'] is not None or (setup_kwds['p'] is not None and (setup_kwds['e'] is not None or setup_kwds['K'] is not None)):
      self.setup(**setup_kwds)

  def set_mode(self,mode):
    #setup basic chain parameters and check conditions are ok
    if mode == 'DEMC':
      #assert self.N>=8, "N must be >=8 for DEMC algorithm"
      if not self.N>=8: raise ValueError("N must be >=8 for DEMC algorithm")
      self.proposal = self.demc_proposal
      if self.cull is None: self.cull = True #cull DEMC by default
    elif mode == 'MH':
      #assert self.N>=2, "N must be >=2 for MH algorithm"
      if not self.N>=2: raise ValueError("N must be >=2 for MH algorithm")
      self.proposal = self.mh_proposal
      if self.cull is None: self.cull = False #do not cull MH by default
    elif mode == 'Gibbs':
      #assert self.N>=2, "N must be >=2 for MH algorithm"
      if not self.N>=2: raise ValueError("N must be >=2 for Gibbs algorithm")
      self.proposal = self.gibbs_proposal
      if self.cull is None: self.cull = False #do not cull Gibbs by default
    else:
      raise ValueError("mode not found. Should be 'DEMC' or 'MH' or [other options to be added]")
  
  @property  
  def filename(self):
    return self._filename
  @filename.setter
  def filename(self,value):
    if value is not None:
      self.autosave = True
    else:
      self.autosave = False
    self._filename = value
      
  #define method to compute single logPosterior evaluation  
  @staticmethod 
  def logPost(p):
    """
    Simple function to compute single logL
    """
    return logP(p)
  
  def map(self,X):
    """
    Replicates central method to compute the logL for current state of the chains.
    This uses the map function and will be parallelised if self.parallel==True
    Might want to modify for small N when not parallised - check for MH algorithm
    """
        
    #compute current state of the samples using map
    return np.array(list(self.map_func(logP,X)))
    
  def run(self,):
    """
    Convenience function to initialise, burnin, and run the chain
    """
    print("not implemented yet")
    pass
    
  def setup(self,X=None,mode=None,N=None,p=None,e=None,K=None,dist='norm',parallel=None,n_burnin=None,burn=0,chain=0,extend=0,cull=None,thin=None,filename=None,**kwargs):
    """
    Initialise the chain positions and compute logP for each
    """
    
    #overwrite any of the defaults if provided in kwargs
    for key in self.defaults.keys():
      if key in kwargs: setattr(self,key,kwargs.get(key))
      
    #and any other parmeters that can be optionally set in __init__ (defaults applied there)
    if N is not None: self.N = N # no of walkers/chains
    if mode is not None: self.mode = mode
    if N is not None or mode is not None:
      self.set_mode(self.mode) #reset mode for either change - checks compatible with N
    #allow some parameters to be reset
    if cull is not None: self.cull = cull
    if thin is not None: self.thin = thin
    if n_burnin is not None: self.n_burnin = n_burnin
    if parallel is not None:
      if parallel:
        self.parallel = True
      else:
        self.parallel = False
        self.map_func = map
    if filename is not None: self.filename = filename

    assert K is not None or e is not None or X is not None, "Must provide samples X or uncertainty as K or e"
    
    if e is not None: e = np.array(e)
    if p is not None: p = np.array(p)
    
    #first define the initial states of the chains
    if X is not None: # X gets priority if given directly
      assert X.shape[0] == self.N, "X is not the correct shape - should be N x n_par"
      self.X = X
    else: #draw samples from distribution
      self.X = self.draw_samples(p=p,e=e,K=K,dist=dist)
        
    #define initial covariance matrix
    if K is not None:
      self.K = K
    elif e is not None:
      self.K = np.diag(e**2)
    else:
      self.K = np.cov(X,rowvar=False)

    #set error vector from cov matrix if not set or else set global value
    if e is None: self.errors = np.sqrt(np.diag(self.K))
    else: self.errors = e
        
    #compute the logP for current X
    if self.parallel: #open/close the pool if running in parallel
      self.pool = multiprocessing.Pool(self.n_p)
      self.map_func = self.pool.map    
    self.XlogP = self.map(self.X)
    if self.parallel:
      self.pool.close()

    print('-' * 100)
    print("{} chain initialised...".format(self.mode))
    print(" No Chains: {}".format(self.N))
    print(" Posterior probability function: {}".format(LogPosterior))
    if burn: print(" Burn in: {} ({} samples)".format(burn,self.N*burn))
    if chain: print(" Chain length: {} ({} samples)".format(chain,self.N*chain))
    print(" Starting marginalised distributions:")
    print(" par = mean += stdev")
    means = self.X.mean(axis=0)
    for q in range(len(self.errors)):
      print(" p[{}] = {} +- {}".format(q,means[q],self.errors[q]))
    
    if np.any(np.isnan(self.XlogP)):
      raise ValueError("logP function returning nans")
    
    if np.any(np.isinf(self.XlogP)):
      print("{}/{} chains are in restricted parameter space (logP = -inf)".format(np.isinf(self.XlogP).sum(),self.N))

    if np.any(self.XlogP-self.XlogP.max()+self.dlogP < 0 ):
      print("{}/{} chains are more than dlogP ({}) from maximum logP".format(np.sum(self.XlogP-self.XlogP.max()+self.dlogP < 0),self.N,self.dlogP))
    
    print('-' * 100)
                
    #reset chains and acceptance etc
    self.chains = None
    self.log_prob = None
    self.Acc = None
    self.g_array = None #reset g_array - required for gibbs
    
    #perform culling
    if self.cull:
      self.recompute_culled(attempts=self.cull_attempts,p=p,e=e,K=K,dist=dist)
      self.redraw_culled()
    
    #run burnin + chains
    if burn>0: self.burn(burn)
    if chain>0:
      self.chain(chain) #run chain
      if extend>0: self.extend(extend) #and any extensions
      return self.p,self.e #return results if chains run
      
  ########################################################################################
  ########################################################################################
   
  def burn(self,n_steps,var_K=None,var_g=None,n_burnin=None):
    """
    Perform burn-in phase. This is same as 'normal' chain but stores output separately
     so it can be discarded and performs updates to the parameters to automatically
     tune the MCMC.
    
    """
    
    if not hasattr(self,'X'):
      raise ValueError("it looks like you haven't run setup yet!")
    
    #allow args to be changed for burnin parameters - some proposal type specific
    if n_burnin is not None:
      self.n_burnin = n_burnin
    if var_K is not None:
      self.var_K = var_K
    if var_g is not None:
      self.var_g = var_g
    
    #define arrays to hold burnin data - these are rewritten each time
    #chain_len x N_chains x p.size
    self.n_steps = n_steps
    self.last_run='burn' #used by some proposal steps
    self.burnin_chunk = n_steps//self.n_burnin #number of steps of chain for each update (ignoring thin)
    if self.burnin_chunk<5: raise ValueError("burnin length is too short: n_steps//n_burnin should be at least 5")
    self.update_steps = [q*self.burnin_chunk for q in range(1,self.n_burnin)]
    self.burntchains = np.zeros((n_steps//self.thin,self.N,self.X.shape[1]))
    self.burntlogP = np.zeros((n_steps//self.thin,self.N))
    self.lastAcc = self.burntAcc = np.full((n_steps,self.N),False) #bool array, lastAcc required for gibbs

    #initial note for 2nd line
    note = "stats (for last chunk of {})".format(self.burnin_chunk)
    #modify note for spefific modes
    if self.mode=='Gibbs' and self.gibbs_ind is not None:
      note = "stats (last chunk of {}), {} logP calls per step (acc for final gibbs step)".format(self.burnin_chunk,np.max(self.gibbs_ind))
    
    if self.parallel: #open the pool if running in parallel
      self.pool = multiprocessing.Pool(self.n_p)
      self.map_func = self.pool.map    
    
    #create random number array for acceptance step
    self.RandNoArr = np.random.rand(n_steps,self.N)
    
    #loop over burnin length and perform updates if required
    pbar_main = tqdm(range(n_steps),position=0,desc="burnin'")
    pbar_updates = tqdm(total=self.n_burnin,position=1,desc=note,bar_format="{desc}{postfix}",leave=True)
    start_time = time.time()
    with pbar_updates:
      for i in pbar_main:      
        update = i in self.update_steps
        X_prop = self.proposal(i,update=update) #create proposal step and optionally update
        XlogP_prop = self.map(X_prop) #compute logP for each
        
        #accept the steps, and update the current state of the chains
        accept = self.RandNoArr[i] < np.exp(XlogP_prop - self.XlogP)
        self.XlogP[accept] = XlogP_prop[accept]
        self.X[accept] = X_prop[accept]
        
        #store the results
        self.burntAcc[i,accept] = True
        #add current posterior and parameters to chain
        if i%self.thin==0: self.burntchains[i//self.thin],self.burntlogP[i//self.thin] = self.X,self.XlogP
        
        if update: #update progress bar with some useful stats
          acc = self.burntAcc[i-self.burnin_chunk:i].sum()*100./self.burntAcc[i-self.burnin_chunk:i].size
          max_gr = max(self.computeGR(X=self.burntchains[(i-self.burnin_chunk)//self.thin:i//self.thin]))
          pbar_updates.set_postfix(acc="{:.1f}%".format(acc),gr="{:.3f}".format(max_gr))
        
    #close the pool if running in parallel
    if self.parallel: self.pool.close()

    #perform final update to proposal steps - useful for non-uniform burnins
    self.proposal(i,update=True,update_only=True)
    
    #print some acceptance stats etc, of whole chain
    #in this recompute acc and max_gr for final chunks
    acc = self.burntAcc[i-self.burnin_chunk:i].sum()*100./self.burntAcc[i-self.burnin_chunk:i].size
    max_gr = max(self.computeGR(X=self.burntchains[(i-self.burnin_chunk)//self.thin:i//self.thin]))
    ts = time.time() - start_time
    print(' Total time: {:.0f}m {:.2f}s'.format(ts // 60., ts % 60.))
    print(' Final chunk acc: {:.2f}%'.format(acc))
    print(' GR (max last burn-in chunk): {:.3f}'.format(max_gr))
    #self.analyse_chains(verbose=verbose)
      
  def chain(self,n_steps,verbose=False,n_updates=10,desc=None):
    """
    Perform main chain.
    verbose - passes to analyse_chains
    n_updates - #of stats updates to progress bar
    
    """
    
    if not hasattr(self,'X'):
      raise ValueError("it looks like you haven't run setup yet!")
    
    #define or extend arrays to hold chain data
    #general shape chain_len x N_chains [x p.size]
    self.n_steps = n_steps
    self.last_run='chain' #used by some proposal steps
    if self.chains is None:
      shift = 0
      if n_steps==0: raise ValueError("Behave yourself! Chains of length 0 not allowed!")
      self.chains = np.zeros((n_steps//self.thin,self.N,self.X.shape[1]))
      self.log_prob = np.zeros((n_steps//self.thin,self.N))
      self.lastAcc = self.Acc = np.full((n_steps,self.N),False) #lastAcc required for gibbs
    else: #extend chains if they already exist
      shift = self.chains.shape[0]
      self.chains = np.vstack([self.chains,np.zeros((n_steps//self.thin,self.N,self.X.shape[1]))])
      self.log_prob = np.vstack([self.log_prob,np.zeros((n_steps//self.thin,self.N))])
      self.Acc =  np.vstack([self.Acc,np.full((n_steps,self.N),False)]) #lastAcc required for gibbs
    
    #initial note for 2nd line
    note = "stats (full chain)"
    #modify note for spefific modes
    if self.mode=='Gibbs' and self.gibbs_ind is not None:
      note = "stats (full chain), {} logP calls per step (acc for final gibbs step)".format(np.max(self.gibbs_ind))

    if self.parallel: #open the pool if running in parallel
      self.pool = multiprocessing.Pool(self.n_p)
      self.map_func = self.pool.map    
    
    #create random number array for acceptance step
    self.RandNoArr = np.random.rand(n_steps,self.N)
        
    #loop over and perform updates to chain
    if desc is None: desc = "chain" if shift==0 else "extending chain"
    pbar_main = tqdm(range(n_steps),position=0,desc=desc)
    pbar_updates = tqdm(total=n_updates,position=1,desc=note,bar_format="{desc}{postfix}",leave=True)
    start_time = time.time()
    with pbar_updates:
      for i in pbar_main:      
        X_prop = self.proposal(i) #create proposal step
        XlogP_prop = self.map(X_prop) #compute logP for each
      
        #accept the steps, and update the current state of the chains
        accept = self.RandNoArr[i] < np.exp(XlogP_prop - self.XlogP)
        self.XlogP[accept] = XlogP_prop[accept]
        self.X[accept] = X_prop[accept]
        
        #store the results
        self.Acc[shift+i,accept] = True
        #add current posterior and parameters to chain
        if i%self.thin==0: self.chains[shift+i//self.thin],self.log_prob[shift+i//self.thin] = self.X,self.XlogP

        if (i+shift)>0 and i%(max(1,n_steps//n_updates))==0: #update progress bar with some useful stats
          acc = self.Acc[:shift+i].sum()*100./self.Acc[:shift+i].size
          max_gr = max(self.computeGR(X=self.chains[:shift+i//self.thin]))
          #pbar.set_postfix(acc="{:.2f}%".format(acc),gr="{:.3f}".format(max_gr))
          pbar_updates.set_postfix(acc="{:.2f}%".format(acc),gr="{:.3f}".format(max_gr))

    #close the pool if running in parallel
    if self.parallel: self.pool.close()
    
    if self.autosave: self.save()

    #print some acceptance stats etc, of whole chain
    ts = time.time() - start_time
    print(' Total time: {:.0f}m {:.2f}s'.format(ts // 60., ts % 60.))
    print(' Final acc: {:.2f}%'.format(self.Acc.sum()*100./self.Acc.size))
    print(' GR (max): {:.3f}'.format(self.computeGR().max()))
    return self.analyse_chains(verbose=verbose)
        
  def save(self,filename=None,verbose=False):
    """Save the current state of the chains"""
    
    if filename is None: filename = self.filename
    np.savez(filename,chains=self.chains,X=self.X,K=self.K,log_prob=self.log_prob,Acc=self.Acc,var_K=self.var_K,var_g=self.var_g)
    if verbose: print(" chains saved to {}".format(self.filename))
    
  def reset(self):
    #reset chains and acceptance etc
    self.chains = None
    self.log_prob = None
    self.Acc = None

  def extend(self,extension_length,n_extensions=None,*args,**kwargs):
    """
    Simple function to extend chains if max gr stat hasn't reached target value
    """

    if n_extensions is not None: self.n_extensions = n_extensions
    
    for n in range(self.n_extensions):
      #test gr stat is ok
      max_gr = self.computeGR().max()
      if max_gr <= self.target_gr:
        if n==0: print('max GR stat already better than target_gr')
        else: print('max GR stat reached target after {}/{} extensions'.format(n,n_extensions))
        return self.p,self.e
      #run extension to the chain
      desc = 'running extension {}/{}'.format(n+1,n_extensions)
      self.chain(extension_length,*args,desc=desc,**kwargs)
    
    #check final state
    max_gr = self.computeGR().max()
    if max_gr <= self.target_gr:
      print('max GR stat reached target after {}/{} extensions'.format(n+1,n_extensions))
    else:
      print('max GR stat not reached target after {}/{} extensions'.format(n+1,n_extensions))    
    
    return self.p,self.e
    
  ########################################################################################
  ########################################################################################

  def draw_samples(self,p=None,e=None,K=None,dist='norm',N=None):
    """
    draw N samples from a given distribution/parameter set
    
    """
    
    if N is None: N = self.N
    
    if p is not None:
      if dist=='norm':
        if K is not None: X = np.random.multivariate_normal(np.array(p),K,N)
        elif e is not None: X = np.random.multivariate_normal(np.array(p),np.diag(e**2),N)
        else: raise ValueError("K or e must be set for dist 'norm'")
      elif dist=='uniform':
        # e takes precedence for uniform dist 
        if e is not None: X = np.array([p + np.random.uniform(-0.5,0.5,p.size) * e for i in range(N)])
        elif K is not None:
          e = np.sqrt(np.diag(K))
          X = np.array([p + np.random.uniform(-0.5,0.5,p.size) * e for i in range(N)])
        else: raise ValueError("e or K must be set for dist 'uniform'")
      else:
        raise ValueError("distribution 'dist' is not set correctly - should be 'norm' or 'uniform'")
    else:
      raise ValueError("to sample from distribution must provide p and ep or K")    
          
    return X

  def recompute_culled(self,attempts=10,p=None,e=None,K=None,dist='norm'):
    """
    cull any discrepant points in X according to dlogP and -np.infs
    """
    
    for n in range(attempts):
      #get index for culling
      cull_index = (self.XlogP==-np.inf) + (self.XlogP < (self.XlogP.max()-self.dlogP))    
      if n==0: n_badpoints = np.sum(cull_index)
      if np.sum(cull_index)==0: break
      #redraw samples
      self.X[cull_index] = self.draw_samples(p=p,e=e,K=K,dist=dist,N=cull_index.sum())
      #and compute new logPs
      self.XlogP[cull_index] = self.map(self.X[cull_index])
    
    #check final state after culling
    cull_index = (self.XlogP==-np.inf) + (self.XlogP < (self.XlogP.max()-self.dlogP))    
    if np.sum(cull_index)>0:
      print("\x1B[3m(warning: after {} attempts {} chains still initiated in restricted prior space!)\x1B[23m".format(attempts,np.sum(cull_index)))
    elif n_badpoints>0:
      print("\x1B[3m(recomputed {} points after max of {} attempts)\x1B[23m".format(n_badpoints,n+1))
        
    return cull_index
    
  def redraw_culled(self):
    """
    Redraw any rejected samples from other chains
    """
    
    cull_index = (self.XlogP==-np.inf) + (self.XlogP < (self.XlogP.max()-self.dlogP))
    ind_good = np.where(~cull_index)[0]
    #assert ind_good.size >= 4, "must have more than 4 good points for culling (have {})!".format(ind_good.size)
    if not ind_good.size >= 4: raise ValueError("must have more than 4 good points for culling (have {})!".format(ind_good.size))
    
    #get random index from good points to replace each bad point
    good_points_ind = np.random.choice(ind_good,cull_index.sum())
    
    #replace the samples and corresponding logP
    self.X[cull_index] = self.X[good_points_ind]
    self.XlogP[cull_index] = self.XlogP[good_points_ind]
    
    if cull_index.sum()>0:
      print("\x1B[3m(culled {}/{} points by redrawing)\x1B[23m".format(cull_index.sum(),self.N))
    
    return cull_index
  
  ########################################################################################
  ########################################################################################

  def demc_proposal(self,i,update=False,update_only=False):
    """
    Perform a single step of the DEMC algorithm.
    If counter is zero, then set up the random variables
    If update is True, then update the parameters - should only be true for burn-in chain
    If update_only then don't compute new step, just adjust the parameters required
      for new chain
    If parallel, then need to draw samples in a slightly different way - ie from other 'independent' chains
    
    """
    
    if i == 0: #initialise any relevant parameters at start of chain
      if self.n_gr is None: self.n_gr = 4
      if self.g is None: self.g = 2.38 / 2. / np.sum(~np.isclose(self.errors,0.))
      self.gamma = np.ones(self.n_steps) * self.g
      if self.n_gamma > 0: self.gamma[self.n_gamma::self.n_gamma] = self.g_glob #set every tenth gamma to one to allow jumps between peaks
      self.R = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.K,(self.n_steps,self.N)) * self.c
      #for n in range(self.N): self.R[:,n,np.where(np.isclose(self.errors,0.))[0]] = 0.
      self.R[...,np.isclose(self.errors,0)] = 0. #ensure array is zeros where e==0

      if not self.parallel: #for normal execution, choose from remainder of chain, ie exclude n
        #get random numbers from 0 to N-2 inclusive
        self.rc = np.random.random((self.n_steps,self.N,self.N-1)).argsort(axis=2)[:,:,:2]
        #where equal to n, set to N-1 to make uniform distribution from remainder
        self.rc[np.equal(self.rc,np.arange(self.N)[np.newaxis,:,np.newaxis])] = self.N-1
      else: #for parallel execution, divide into two groups, and simply pick from other group
        self.rc = np.random.random((self.n_steps,self.N,self.N//2)).argsort(axis=2)[:,:,:2]
        self.rc[:,:self.N//2] += self.N//2
        #print("\x1B[3m(running in parallel with {} cores)\x1B[23m".format(n_p))

    elif update: #perform update step, only done for i>0, and for burnin
      if self.var_K: #update convariance matrix
        #print(i,"recomputing K")
        sl = slice( (i-self.burnin_chunk)//self.thin,i//self.thin ) #take the last burnin chunk
        #recompute K and random arrays - will only ever be done on the burnin-chains
        #use a 'damped' update, i.e. weighted combination with old K
        #(should double check if there's a better way to do this)
        #print('subtract mean first to avoid rounding errors?')
        #self.K = (self.K + 4.*np.cov(self.burntchains[sl].reshape(-1,self.X.shape[1]),rowvar=0))/5.
        self.K = (self.K + 4.*np.cov(self.burntchains[sl].reshape(-1,self.X.shape[1])-self.burntchains.mean(axis=(0,1)),rowvar=0))/5.
        self.K[np.where(np.isclose(self.errors,0.))],self.K[:,np.where(np.isclose(self.errors,0.))] = 0.,0. #reset error=0. values to 0. to avoid numerical errors
        if not update_only: #only recompute the random variables if there's more of the chain the compute!
          #could restrict to only next chunk, but little performance hit - maybe later
          self.R[i:] = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.K,(self.n_steps-i,self.N)) * self.c    
          self.R[...,np.isclose(self.errors,0)] = 0. #ensure array is zeros where e==0
          
      if self.var_g:
        #print(i,"recomputing gamma")
        sl = slice(i-self.burnin_chunk,i)
        #target_acc = 0.234 #target acceptance
        #rescale gamma for target acceptance
        acc = self.burntAcc[sl].sum() / self.burntAcc[sl].size
        #print('acc =',acc)
        self.g *= (1./self.target_acc) * min(0.8,max(0.1,acc))
        if not update_only: #only recompute the random variables if there's more of the chain the compute!
          self.gamma[i:] = self.g
          #and reset global gamma values
          if self.n_gamma > 0: self.gamma[self.n_gamma::self.n_gamma] = self.g_glob
        #print('g =',self.g)
      if update_only:
        #print("update step only - returning")
        return 1
    
    #generate proposal step from current set of chains and return
    X_prop = self.X + self.gamma[i] * (self.X[self.rc[i,:,0]] - self.X[self.rc[i,:,1]]) + self.R[i]
    return X_prop
    
  def mh_proposal(self,i,update=False,update_only=False):
    """
    Metropolis-Hastings step
    By default will use global covariance and individual 'global' stepsizes for each step
    Can use orthogonal steps (mcmc.orth=True) - just sets covariance to zero except for diagonal
    Can use independent covariance (mcmc.global_K=False)
    
    """
    
    if i == 0: #initialise any relevant parameters at start of chain
      if self.n_gr is None: self.n_gr = self.N
      if self.g is None: self.g = 2.4**2 / np.sum(~np.isclose(self.errors,0.)) * np.ones(self.N)
      if self.global_K:
        self.R = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.K,(self.n_steps,self.N)) * self.g[...,np.newaxis]
      else:
        if not hasattr(self,'Kn'): self.Kn = [self.K for n in range(self.N)]
        self.R = np.empty((self.n_steps,self.N,self.X.shape[1]))
        for n in range(self.N): #loop over Kn and update each chain separately
          self.R[:,n] = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.Kn[n],(self.n_steps)) * self.g[n]
      self.R[...,np.isclose(self.errors,0)] = 0. #ensure array is zeros where e==0
      
    elif update: #perform update step, only done for i>0 and for burnin
      if self.var_K: #update convariance matrix
        sl = slice( (i-self.burnin_chunk)//self.thin,i//self.thin ) #take the last burnin chunk
        
        #get new covariance matrix from samples
        if self.global_K:
          K = np.cov(self.burntchains[sl].reshape(-1,self.X.shape[1])-self.burntchains.mean(axis=(0,1)),rowvar=0)
          if self.orth: K = np.diag(np.diag(K)) #use diagonal components only - same as taking variance in each dimension
          self.K = (self.K + 4.*K)/5.
          self.K[np.where(np.isclose(self.errors,0.))],self.K[:,np.where(np.isclose(self.errors,0.))] = 0.,0. #reset error=0. values to 0. to avoid numerical errors
          #recompute the random steps
          if not update_only: #only recompute the random variables if there's more of the chain the compute!
            self.R[i:] = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.K,(self.n_steps-i,self.N)) * self.g[...,np.newaxis]
            self.R[i:,:,np.isclose(self.errors,0)] = 0. #ensure array is zeros where e==0
        else:
          Kn = [np.cov(self.burntchains[sl,n]-self.burntchains[:,n].mean(axis=0),rowvar=0) for n in range(self.N)]
          if self.orth: self.Kn = [np.diag(np.diag(k)) for k in self.Kn] #force each Kn to be diagonal
          self.Kn = [(skn + 4.*kn)/5. for skn,kn in zip(self.Kn,Kn)] #update each Kn with damping
          for n in range(self.N):
            self.Kn[n][np.where(np.isclose(self.errors,0.))],self.Kn[n][:,np.where(np.isclose(self.errors,0.))] = 0.,0. #reset error=0. values to 0. to avoid numerical errors
          if not update_only:
            for n in range(self.N): #loop over Kn and update each chain separately
              self.R[i:,n] = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.Kn[n],(self.n_steps-i)) * self.g[n]
            self.R[i:,:,np.isclose(self.errors,0)] = 0. #ensure array is zeros where e==0
          #compute new K separately for each chain - will need to overwrite or create new K
          #raise ValueError("Independent K for each chain not yet supported")

      if self.var_g:
        sl = slice(i-self.burnin_chunk,i) #no thinning applied to acceptance array
        #target_acc = 0.234 #target acceptance
        #rescale G for target acceptance in each chain
        acc_array = self.burntAcc[sl].sum(axis=0) / self.burntAcc[sl].shape[0]
        g_scaling = (1./self.target_acc) * np.minimum(0.9,np.maximum(0.1,acc_array))
        self.g *= g_scaling
        if not update_only:
          self.R[i:] *= g_scaling[...,np.newaxis] # important to update R as well as g/K may not be updated together
      if update_only:
        return 1
    
    #generate proposal step from current set of chains and return
    X_prop = self.X + self.R[i]
    return X_prop
   
  def gibbs_proposal(self,i,update=False,update_only=False):
    """
    gibbs_index = [0,1,0,1,2,2,3] will vary parameters 1, 2 then 3 in turn, evaluating
    the log posterior at each and accepting them according to the MH rule.
    
    """
    
    if i == 0: #initialise any relevant parameters at start of chain
      #first check gibbs index is ok
      if self.gibbs_ind is None: raise ValueError("gibbs requires 'gibbs_ind' to be set")
      self.gibbs_ind = np.array(self.gibbs_ind)
      if not self.gibbs_ind.size==self.errors.size: raise ValueError("gibbs_ind must be same length as pars")
      n_gibbs = self.n_gibbs = self.gibbs_ind.max()
      if self.n_gibbs<2: raise ValueError("gibbs requires at least 2 blocks - use MH instead")
      if not np.all([i in self.gibbs_ind for i in range(1,n_gibbs+1)]): raise ValueError("gibbs_ind must be continuous?/no gaps?")
      
      if self.n_gr is None: self.n_gr = self.N

      #create new random number array for internal acceptance step
      #don't need final gibbs step - that will be handled by main chain
      self.gibbsRandNoArr = np.random.rand(self.RandNoArr.shape[0],n_gibbs-1,self.N)
      self.gibbsAcc = np.full((self.RandNoArr.shape[0],n_gibbs-1,self.N),False) #bool array
      #point MainAcc to current chain? Not needed as updates only done with bunrtAcc
      #self.mainAcc = self.Acc if self.last_run == 'chain' else self.burntAcc
      
      #first need g for each gibbs_step and chain - these will be updated (array includes zeros)
      #N x ngibbs+1
      #self.g = np.array([[(2.4**2/(self.gibbs_ind==q).sum()) if q>0 else 0 for q in range(self.n_gibbs+1)] for n in range(self.N)])
      if self.g_array is None:
        self.g = np.outer(np.ones(self.N),[(2.4**2/(self.gibbs_ind==q).sum()) if q>0 else 0 for q in range(n_gibbs+1)])
        #then expand to match parameter array and chain - required for multiplication into R
        self.g_array = np.array([[self.g[n][q] for q in self.gibbs_ind] for n in range(self.N)])
        #finally get list of indexes for fast indexing of each gibbs set
        self.gibbs_indices = [self.gibbs_ind==q for q in range(1,n_gibbs+1)]
        
      #covariance matrix can be treated in exactly the same way as MH
      #with the exception of changing how g multiplies into the random array R
      if self.global_K:
        self.R = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.K,(self.n_steps,self.N)) * self.g_array
      else:
        if not hasattr(self,'Kn'): self.Kn = [self.K for n in range(self.N)]
        self.R = np.empty((self.n_steps,self.N,self.X.shape[1]))
        for n in range(self.N): #loop over Kn and update each chain separately
          self.R[:,n] = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.Kn[n],(self.n_steps)) * self.g_array[n]
      self.R[...,np.isclose(self.errors,0)] = 0. #ensure array is zeros where e==0

    elif update: #perform update step, only done for i>0 and for burnin
      if self.var_K: #update convariance matrix
        sl = slice( (i-self.burnin_chunk)//self.thin,i//self.thin ) #take the last burnin chunk
        
        #get new covariance matrix from samples
        if self.global_K:
          K = np.cov(self.burntchains[sl].reshape(-1,self.X.shape[1])-self.burntchains.mean(axis=(0,1)),rowvar=0)
          if self.orth: K = np.diag(np.diag(K)) #use diagonal components only - same as taking variance in each dimension
          self.K = (self.K + 4.*K)/5.
          self.K[np.where(np.isclose(self.errors,0.))],self.K[:,np.where(np.isclose(self.errors,0.))] = 0.,0. #reset error=0. values to 0. to avoid numerical errors
          #recompute the random steps
          if not update_only: #only recompute the random variables if there's more of the chain the compute!
            self.R[i:] = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.K,(self.n_steps-i,self.N)) * self.g_array
            self.R[i:,:,np.isclose(self.errors,0)] = 0. #ensure array is zeros where e==0
        else:
          Kn = [np.cov(self.burntchains[sl,n]-self.burntchains[:,n].mean(axis=0),rowvar=0) for n in range(self.N)]
          if self.orth: self.Kn = [np.diag(np.diag(k)) for k in self.Kn] #force each Kn to be diagonal
          self.Kn = [(skn + 4.*kn)/5. for skn,kn in zip(self.Kn,Kn)] #update each Kn with damping
          for n in range(self.N):
            self.Kn[n][np.where(np.isclose(self.errors,0.))],self.Kn[n][:,np.where(np.isclose(self.errors,0.))] = 0.,0. #reset error=0. values to 0. to avoid numerical errors
          if not update_only:
            for n in range(self.N): #loop over Kn and update each chain separately
              self.R[i:,n] = np.random.multivariate_normal(np.zeros(self.X.shape[1]),self.Kn[n],(self.n_steps-i)) * self.g_array[n]
            self.R[i:,:,np.isclose(self.errors,0)] = 0. #ensure array is zeros where e==0
          #compute new K separately for each chain - will need to overwrite or create new K
          #raise ValueError("Independent K for each chain not yet supported")

      if self.var_g:
        sl = slice(i-self.burnin_chunk,i) #no thinning applied to acceptance array        
        #rescale G for target acceptance in each chain and each set of gibbs updates
        for q in range(self.n_gibbs):
          #for final step
          if q==self.n_gibbs-1: acc_array = self.burntAcc[sl].sum(axis=0) / self.burntAcc[sl].shape[0]
          #for the others
          else: acc_array = self.gibbsAcc[sl,q].sum(axis=0) / self.gibbsAcc[sl,q].shape[0]
          #now get scaling for each gibbs_step - for every N
          g_scaling = (1./self.target_acc) * np.minimum(0.9,np.maximum(0.1,acc_array))          
          #careful to take into account that zero is also (and always) stored in self.g
          self.g[:,q+1] *= g_scaling #scale each g value - just a column for each gibbs_set
          #and assign to the g_array - now corresponding to the full parameter array
          self.g_array[:,self.gibbs_indices[q]] = self.g[:,q+1][:,np.newaxis] #*= g_scaling
          
#           print('checking gibbs')
#           print(q)
#           print(self.g[:,q+1])
#           print(self.gibbs_indices[q])
#           print(acc_array)
#           print(g_scaling)
#           
          #finally apply scaling to the R values if chain isn't finished
          if not update_only:
            #only apply to current gibbs step!
            self.R[i:,:,self.gibbs_indices[q]] *= g_scaling[:,np.newaxis] # important to update R as well as g/K may not be updated together
          
      if update_only:
        return 1
    
    #loop over first n_gibbs-1 steps within the proposal step
    for q in range(self.n_gibbs):
      
      #add on each set of random gibbs arrays in turn
      X_prop_gibbs = np.copy(self.X)
      X_prop_gibbs[:,self.gibbs_indices[q]] += self.R[i][:,self.gibbs_indices[q]]
      #if the last step, send back to normal chain
      if q==self.n_gibbs-1: return X_prop_gibbs
      #else evaluate within func
      XlogP_prop_gibbs = self.map(X_prop_gibbs) #compute logP for each      
#     #accept the steps, and update the current state of the chains
      accept = self.gibbsRandNoArr[i][q] < np.exp(XlogP_prop_gibbs - self.XlogP)
      self.XlogP[accept] = XlogP_prop_gibbs[accept]
      self.X[accept] = X_prop_gibbs[accept]
      #update local acceptance chain
      #print('carefully check if this works for i,q,boolean array index')
      #looks ok for 2 ints + accept 
      self.gibbsAcc[i,q,accept] = True

    print("Something went wrong in gibbs_proposal - you shouldn't be here!")
    
  ########################################################################################
  ########################################################################################

  def chains_reshaped(self,N):
    """
    Simple func to reshape chains array into pseudo chains of different N
    """
    assert self.N%N == 0 and N>0, ValueError("N must be a positive divisor of self.N")
    
    if N==1: reshaped = self.chains.reshape(-1,self.chains.shape[-1])
    else: reshaped = self.chains.reshape(-1,N,self.chains.shape[-1])
    
    return reshaped
  
  def samples(self,Nsamples=None):
    """
    Simple function to return samples from the current chains.
    If Nsamples is None, returns all samples, else picks Nsamples of them at random
    """
    
    X = self.chains_reshaped(1)
    
    if Nsamples is None: return X
    else:
      ind = np.random.randint(0,X[:,0].size,Nsamples)
      return X[ind]
  
  def computeGR(self,X=None,conv=0,n_gr=None):
    """
    Compute the Gelman and Rubin statistic and errors for all variable parameters
     after optionally reshaping chains.
    Convergence is given in terms of original chain length
    """
    
    if n_gr is None: n_gr=self.n_gr
    
    assert self.N%n_gr == 0 and n_gr>1, ValueError("n_gr must be a divisor of N and at least 2")

    if X is None: #by default use chains and reshape into n_gr
      X = self.chains_reshaped(n_gr)
    else: #or else reshape the given chain
      X = X.reshape(-1,n_gr,X.shape[-1])
    conv = conv * self.N // n_gr #rescale convergence to match new shape
    L = X.shape[0]-conv #length of individual chain
    assert conv>=0, ValueError("conv must be >=0")
    assert L>0, ValueError("conv set is too high for chain of length {}".format(X.shape[0]))
    X = X[conv:] #filter out steps before convergence

    #compute only for variables
    p = np.where(X.std(axis=(0,1))>0)[0]
    grs = np.zeros(self.errors.size)
    
    for i in p:
    
      #get mean and variance for the individual chains
      mean = X[...,i].mean(axis=0)
      var = X[...,i].var(axis=0)
          
      #and calculate the GR stat
      W = var.mean(dtype=np.float64) #mean of the variances
      B = mean.var(dtype=np.float64) #variance of the means
      grs[i] = np.sqrt((((L-1.)/L)*W + B) / W) #GR stat
  
    return grs
  
  def analyse_chains(self,X=None,conv=0,verbose=True,logP_max=None):
    """
    Simple function to get results from current chains
    """
    
    #get a flattened chain
    if X is None: #by default use chains and reshape into n_gr
      X = self.chains_reshaped(1)
    else: #or else reshape the given chain
      X = X.reshape(-1,1,X.shape[-1])
    conv = conv * self.N #rescale convergence to match new shape
    X = X[conv:] # get rid of points before convergence
    assert X.shape[0]>0, ValueError("conv set is too high for chain of length {}".format(X.shape[0]))
    
    self.means = means = X.mean(axis=0)
    self.medians = medians = np.median(X,axis=0)
    self.stdevs = stdevs = X.std(axis=0)
    #get assymetric errors
    self.lower_lims,self.medians,self.upper_lims = np.percentile(X,[15.9,50,100-15.9],axis=0)
    pos_errs = self.upper_lims - medians
    neg_errs = self.medians - self.lower_lims
    grs = self.computeGR(conv=conv//self.N)
    
    if verbose:
      print('-' * 100)
      print ("MCMC Marginalised distributions:")
      print (" par = mean +- stdev [med +err -err]: GR")
      for i in range(self.means.size):
#        print (" p[%d] = %.7f +- %.7f [%.7f +%.7f -%.7f]: GR = %.4f" % (i,mean[i],gauss_err[i],median[i],pos_err[i],neg_err[i],GR[i]))
        print (" p[{}] = {:.7f} +- {:.7f} [{:.7f} +{:.7f} -{:.7f}]: GR = {:.4f}".format(i,means[i],stdevs[i],medians[i],pos_errs[i],neg_errs[i],grs[i]))
     
      #calculate evidence approximations
      K = np.cov(X-means,rowvar=0) #get covariance - better to mean subtract first to avoid rounding errors
      if logP_max is None: logP_max = np.max(self.log_prob) #get maximum posterior
      
      #compress K where 
      var_par = np.diag(K)>0
      Ks = K.compress(var_par,axis=0)
      Ks = Ks.compress(var_par,axis=1)
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
      print('-' * 100)

    return means,stdevs
  
  #define some simple properties to compute from the chains
  @property
  def cov(self):
    return np.cov(self.chains_reshaped(1),rowvar=0)    
  @property
  def p(self):
    return np.mean(self.chains_reshaped(1),axis=0)
  @property
  def e(self):
    return np.std(self.chains_reshaped(1),axis=0)
  @property
  def results(self):
    return np.mean(self.chains_reshaped(1),axis=0),np.std(self.chains_reshaped(1),axis=0)
  grs = property(computeGR)
  
##########################################################################################
##########################################################################################
