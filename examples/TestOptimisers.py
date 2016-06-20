#!/usr/bin/env python

import pylab
import numpy as np
import Infer
import scipy.optimize as opt
import MyFuncs as MF
import os

#define non-linear mean function
def mf(p,time):
  return (p[0] + p[1] + p[1]*time + p[2]*time**2 + p[3]*np.sin(2*np.pi*p[4]*time) + p[5]*np.cos(2*np.pi*p[6]*time))

#define merit function for levenberg marquardt
def LM_ErrFunc(par,func,func_args,y,err):
  return (y - func(par,func_args)) / err

##########################################################################################
#generate some fake data
time = np.linspace(0,1,500)
par = [-40,40,-40,3,2,7,4]
par_noise = 1.
par_guess = np.random.normal(par,par_noise) #add 'noise' to the guess parameters
epar = [par_noise,]*7 + [0.0,]
fp = [0,0,0,0,0,0,0,1]
wn = 0.6
f = mf(par,time) + np.random.normal(0,wn,time.size)

##########################################################################################

#fit with levenburg marquardt
NM_temp = Infer.Optimise(MF.LogLikelihood_iid_mf,np.concatenate([par_guess,[wn,]]),(mf,time,f),fixed=fp)
LM1 = opt.leastsq(LM_ErrFunc,NM_temp[:-1],args=(mf,time,f,1),full_output=1)
rescale1 = (f-mf(LM1[0],time)).std()
LM1_par = LM1[0]
LM1_epar = np.sqrt(np.diag(LM1[1])) * rescale1

#fit with least squares - again a levenberg marquardt method (trust region is similar idea)
LM2 = opt.least_squares(LM_ErrFunc,par_guess,args=(mf,time,f,1),method='trf',verbose=1)
rescale2 = (f-mf(LM2.x,time)).std()
H = np.dot(LM2.jac.T,LM2.jac)
C = np.linalg.inv(H)
LM2_par = LM2.x
LM2_epar = np.sqrt(np.diag(C)) * rescale2

#fit with Infer.LevMar method
LM3_par,LM3_epar,R,K,logE = Infer.LevMar2(mf,par_guess,(time,),f,fixed=~(np.array(epar[:-1])>0))

#fit with simple MCMC
lims=[1000,20000,5]
Infer.MCMC_N(MF.LogLikelihood_iid_mf,np.concatenate([par_guess,[wn,]]),(mf,time,f),30000,epar,adapt_limits=lims, glob_limits=lims,N=2)
MCMC_par,MCMC_epar = Infer.AnalyseChains(20000,n_chains=2)
os.system('rm MCMC_chain_?.npy')

#fit with nelder mead
NM_par = Infer.Optimise(MF.LogLikelihood_iid_mf,np.concatenate([par_guess,[wn,]]),(mf,time,f),fixed=fp)

#fit with brute force/Infer
BR_par = Infer.Brute(MF.LogLikelihood_iid_mf,np.concatenate([par_guess,[wn,]]),(mf,time,f),epar,Niter=2000,verbose=False)
#BRUTE2 = Infer.Optimise(MF.LogLikelihood_iid_mf,BRUTE,(mf,time,f),fixed=[0,0,0,0,0,1])
BRLM_par,BRLM_epar,R,K,logE = Infer.LevMar2(mf,BR_par,(time,),f,fixed=~(np.array(epar[:-1])>0))

#fit with differential evolution algorithm and then LM
DE_par = Infer.DifferentialEvolution(MF.LogLikelihood_iid_mf,np.concatenate([par_guess,[wn,]]),(mf,time,f),epar)
DELM_par,DELM_epar,R,K,logE = Infer.LevMar2(mf,DE_par,(time,),f,fixed=~(np.array(epar[:-1])>0))

#------------------------------

#plot results
pylab.figure(figsize=(8,11))
pylab.subplot(711)
pylab.plot(time,f,'k.')
pylab.plot(time,mf(par,time),'r-')
pylab.plot(time,mf(par_guess,time),'b-')
pylab.plot(time,mf(LM1_par,time),'g-',lw=1)
pylab.ylabel('LM1')

pylab.subplot(712)
pylab.plot(time,f,'k.')
pylab.plot(time,mf(par,time),'r-')
pylab.plot(time,mf(par_guess,time),'b-')
pylab.plot(time,mf(LM2_par,time),'g-',lw=1)
pylab.ylabel('LM2')

pylab.subplot(713)
pylab.plot(time,f,'k.')
pylab.plot(time,mf(par,time),'r-')
pylab.plot(time,mf(par_guess,time),'b-')
pylab.plot(time,mf(LM3_par,time),'g-',lw=1)
pylab.ylabel('LM3')

pylab.subplot(714)
pylab.plot(time,f,'k.')
pylab.plot(time,mf(par,time),'r-')
pylab.plot(time,mf(par_guess,time),'b-')
pylab.plot(time,mf(MCMC_par,time),'g-',lw=1)
pylab.ylabel('MCMC')

pylab.subplot(715)
pylab.plot(time,f,'k.')
pylab.plot(time,mf(par,time),'r-')
pylab.plot(time,mf(par_guess,time),'b-')
pylab.plot(time,mf(NM_par,time),'g-',lw=1)
pylab.ylabel('NM')

pylab.subplot(716)
pylab.plot(time,f,'k.')
pylab.plot(time,mf(par,time),'r-')
pylab.plot(time,mf(par_guess,time),'b-')
pylab.plot(time,mf(BR_par,time),'c-',lw=1)
pylab.plot(time,mf(BRLM_par,time),'g-',lw=1)
pylab.ylabel('BR')

pylab.subplot(717)
pylab.plot(time,f,'k.')
pylab.plot(time,mf(par,time),'r-')
pylab.plot(time,mf(par_guess,time),'b-')
pylab.plot(time,mf(DE_par,time),'c-',lw=1)
pylab.plot(time,mf(DELM_par,time),'g-',lw=1)
pylab.ylabel('DE')
pylab.draw()

print "RMSs:"
print ((f-mf(LM1_par,time))**2).sum() / time.size
print ((f-mf(LM2_par,time))**2).sum() / time.size
print ((f-mf(LM3_par,time))**2).sum() / time.size
print ((f-mf(MCMC_par,time))**2).sum() / time.size
print ((f-mf(NM_par,time))**2).sum() / time.size
print ((f-mf(BR_par,time))**2).sum() / time.size
print ((f-mf(BRLM_par,time))**2).sum() / time.size
print ((f-mf(DE_par,time))**2).sum() / time.size
print ((f-mf(DELM_par,time))**2).sum() / time.size

print "Pars"
print LM1_par
print LM2_par
print LM3_par
print NM_par
print MCMC_par
print BRLM_par
print DE_par
print DELM_par

print "Errors"
print LM1_epar
print LM2_epar
print LM3_epar
print MCMC_epar
print BRLM_epar
print DELM_epar

raw_input()
