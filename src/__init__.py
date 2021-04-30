"""
Infer module - useful tools for inference. MCMC methods, optimisation, Importance
Sampling, GPs, etc, with special attention to make the code fast but adaptable. Much of it
is tested against equivalent C modules - it is almost as fast but much more flexible.

Neale Gibson
ngibson@eso.org
nealegibby@gmail.com

leastsqbound.py was adapted from https://github.com/jjhelmus/leastsqbound-scipy/
- It is Copyright (c) 2012 Jonathan J. Helmussee, see file for full license

"""

#set multiprocessing behaviour for python3.8 or above
import sys
if sys.version_info >= (3, 8):
  try:
    import multiprocessing
    multiprocessing.set_start_method("fork")
  except:
    pass

from .MCMC import MCMC
from .MCMC_SimN import MCMC_N
from .MCMC_utils import *
from .ImportanceSampling import *
from .Conditionals import *
from .Optimiser import *
from .BruteForce import *
from .DifferentialEvolution import *
from .LevenbergMarquardt import *
#from LevenbergMarquardt2 import *

from .MCMC_BGibbs import *
from .AffInv_MCMC import *
from .DEMCMC import *
from . import TestParallelSpeed

# from .InferGP import GP
# from .InferMGP import MGP
# 
# from .GPUtils import *
# from . import GPRegression as GPR
# from . import GPCovarianceMatrix as GPC
# 
# from .GPKernelFunctions import *
# from .GPPeriodicKernelFunctions import *
# 
# from .GPToeplitz import *
