
import numpy as np
import time
from scipy.optimize import differential_evolution

##########################################################################################

def DifferentialEvolution(LogLikelihood,par,func_args,epar,type='max',Nsig=3,verbose=True,**kwargs):
  """
  Function wrapper to find the maximum (or min) of a function using the scipy differential
  evolution function
  
  LogLikelihood - function to optimise, of the form func(parameters, func_args). Doesn't
    need to be a log likelihood of course - just that's what I use it for!
  par - array of parameters to the func to optimise
  func_args - additional arguments to the func - usually as a tuple
  epar - array of parameter errors
  Nsig - no of sigma from errbars and par to set bounds
  type - either max or min to optimise the funciton
  
  """
  
  #first set bounds for search range - use Nsig away from guess par
  bounds = [(p-Nsig*ep,p+Nsig*ep) for p,ep in zip(par,epar)]
  
  #redefine for fixed parameters
  if verbose:
    print "Differential Evolution parameter ranges:"
    for i in range(len(par)):
      print " p[{}] => {}".format(i,bounds[i])
  
  assert type == 'max' or type == 'min', "type must be max or min"
  if type == 'max': OptFunc = NegFunc
  elif type == 'min': OptFunc = PosFunc
  
  #run the brute force algorithm, without finishing algorithm
  DE = differential_evolution(OptFunc,bounds,args=(LogLikelihood,func_args),polish=False,**kwargs)
  
  #print out results
  if verbose: print "No of function evaluations = {}".format(DE.nfev)
  if verbose: print "DE {} grid point @ {}".format(type,DE.x)
  
  #return the optimised position
  return DE.x
  
##########################################################################################
#define wrappers to return the log likelihood for min or max in same way

def NegFunc(par,func,func_args,**kwargs):
  #return negative of function
  return -func(par,*func_args,**kwargs)

def PosFunc(par,func,func_args,**kwargs):
  #return negative of function
  return func(par,*func_args,**kwargs)

##########################################################################################
#create function aliases
DiffEvol = DifferentialEvolution
DE = DifferentialEvolution

##########################################################################################
