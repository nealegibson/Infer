
import numpy as np
import time
from scipy.optimize import fmin,brute,fmin_cg,fmin_powell,fmin_bfgs,fmin_l_bfgs_b

##########################################################################################
def Brute(LogLikelihood,par,func_args,epar,type='max',Nsig=3,Niter=1000,verbose=True):
  """
  Function wrapper to find the maximum (or min) of a function using the scipy brute
  force function
  
  LogLikelihood - function to optimise, of the form func(parameters, func_args). Doesn't
    need to be a log likelihood of course - just that's what I use it for!
  par - array of parameters to the func to optimise
  func_args - additional arguments to the func - usually as a tuple
  epar - array of parameter errors
  Nsig - no of sigma to search from the error bars
  Niter - approximate number of iterations - rounded up to next integer for no of points per variable
  type - either max or min to optimise the funciton
  
  """
  
  #first get fixed parameters
  fixed = ~(np.array(epar) > 0) * 1
  Nvar = fixed.size - fixed.sum() #no of variable parameters
  
  #get number of points to slice each variable parameter - round up!
  #first round to 5 dp to avoid machine prec errors
  Npoints = np.ceil(np.round(Niter ** (1./Nvar),decimals=5))
  
  #define parameter ranges as slice objects
  delta = 0.000001 #make delta par for small increment
  par_ranges = [slice(p-Nsig*e,p+(Nsig+delta)*e,2*Nsig*e/(Npoints-1.)) for p,e in zip(par,epar)]
  
  #redefine for fixed parameters
  if verbose: print ("Brute parameter ranges: ({:d} evaluations)".format(int(Npoints**Nvar)))
  for i,f in enumerate(fixed):
    if f == 1: #redefine slice so only actual par is taken
      par_ranges[i] = slice(par[i],par[i]+delta,0.1)
    if verbose: print (" p[{}] => {}".format(i,np.r_[par_ranges[i]]))
  
  assert type == 'max' or type == 'min', "type must be max or min"
  if type == 'max': OptFunc = NegFunc
  elif type == 'min': OptFunc = PosFunc
  
  #run the brute force algorithm, without finishing algorithm
  B = brute(OptFunc,par_ranges,args=(LogLikelihood,func_args),full_output=1,finish=None)
  
  #print out results
  if verbose: print ("Brute force {} grid point @ {}\n".format(type,B[0]))

  #return the optimised position
  return B[0]
  
##########################################################################################
#define wrappers to return the log likelihood for min or max in same way

def NegFunc(par,func,func_args,**kwargs):
  #return negative of function
  return -func(par,*func_args,**kwargs)

def PosFunc(par,func,func_args,**kwargs):
  #return negative of function
  return func(par,*func_args,**kwargs)

##########################################################################################
