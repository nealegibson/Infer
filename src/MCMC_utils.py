"""
MCMC utils for analysing Infer.MCMC output.
"""

import numpy as np
import pylab
import scipy.ndimage as ndimage

###############################################################################

def AnalyseChains(conv_length,n_chains=None,chain_filenames=None,log_proposal=False,return_GR=False,return_assym=False,N_obs=None,return_ev=False,return_BIC=False,return_AIC=False):
  """
  Get mean, median plus uncertainties from the MCMC chains, and calculate GR statistic
  for multiple chains.
  
  """
  
  #get chain names tuple
  if n_chains == None and chain_filenames == None:
    chain_filenames=("MCMC_chain",)  
  if n_chains != None:
    chain_filenames = ["MCMC_chain_%d" % i for i in range(1,n_chains+1)]
  if type(chain_filenames) is str:
    chain_filenames = [chain_filenames,]
  
  #get number of parmaeters
  #no_pars = len(open(chain_filenames[0],'r').readline().split())-1
  no_pars = np.load(chain_filenames[0]+'.npy')[0].size-1
  
  #set empty arrays for the mean and gaussian errors (for returning)
  mean = np.empty(no_pars)
  gauss_err = np.empty(no_pars)
  GR = np.empty(no_pars)
  pos_err = np.empty(no_pars)
  neg_err = np.empty(no_pars)
  
  print ("MCMC Marginalised distributions:")
  print (" par = mean gauss_err [med +err -err]: GR")
  for i in range(no_pars): #loop over parameters and get parameters, errors, and GR statistic
    mean[i],med,gauss_err[i],pos_err[i],neg_err[i],GR[i] = GelRub(chain_filenames,i+1,conv_length)
#    log_p_str = u", \u0394logp = %f" % ((np.log10(mean[i]+pos_err) - np.log10(mean[i]-neg_err))/2.) if log_proposal else ""
#    print u" p[%d] = %.7f \u00B1 %.7f [%.7f +%.7f -%.7f]: GR = %.4f%s" % (i,mean[i],gauss_err[i],med,pos_err,neg_err,GR,log_p_str)
    log_p_str = ", dlogp = %f" % ((np.log10(mean[i]+pos_err[i]) - np.log10(mean[i]-neg_err[i]))/2.) if log_proposal else ""
    print (" p[%d] = %.7f +- %.7f [%.7f +%.7f -%.7f]: GR = %.4f%s" % (i,mean[i],gauss_err[i],med,pos_err[i],neg_err[i],GR[i],log_p_str))
  
  #return the Evidence approximation from Gaussian assumption
  #merge files into large matrix
  X = np.load(chain_filenames[0]+'.npy')[conv_length:] #read in data file  
  for i in range(1,len(chain_filenames)):
    X = np.concatenate((X,np.load(chain_filenames[i]+'.npy')[conv_length:]))
  #get the mean and covariance matrix
  m = X[:,1:].mean(axis=0)
  K = np.cov(X[:,1:]-m,rowvar=0) #better to mean subtract first to avoid rounding errors
  #get max likelihood
  logP_max = np.nanmax(X[:,0])
  #first must compress the covariance matrix as some parameters are fixed!
  var_par = np.diag(K)>0
  Ks = K.compress(var_par,axis=0);Ks = Ks.compress(var_par,axis=1)
  D = np.diag(Ks).size
#  ms = m[var_par]
  sign,logdetK = np.linalg.slogdet( 2*np.pi*Ks ) # get log determinant
  logE = logP_max + 0.5 * logdetK #get evidence approximation based on Gaussian assumption
  print ("Gaussian Evidence approx:")
  print (" log ML =", logP_max)
  print (" log E =", logE)
  if not N_obs:
    logE_BIC = logP_max
    print (" log E (BIC) = log ML - D/2.*np.log(N) =", logP_max, "- {}/2.*np.log(N)".format(D))
  else:
    logE_BIC = logP_max - D/2.*np.log(N_obs)
    print (" log E (BIC) = log ML - D/2.*np.log(N) =", logE_BIC, "(D = {}, N = {})".format(D,N_obs))
  
  if not N_obs:
    logE_AIC = logP_max - D
    print (" log E (AIC) = log ML - D =", logE_AIC, "(D = {})".format(D))
  else:
    logE_AIC = logP_max - D * N_obs / (N_obs-D-1.)
    print (" log E (AIC) = log ML - DN/(N-D-1) =", logE_AIC, "(D = {}, N = {})".format(D,N_obs))
  
  ret_list = [mean,gauss_err]
  if return_GR: ret_list.append(GR)
  if return_assym: ret_list.append(neg_err);ret_list.append(pos_err)
  if return_ev: ret_list.append(logE)
  if return_BIC: ret_list.append(logE_BIC)
  if return_AIC: ret_list.append(logE_AIC)
  return ret_list
  
#   if return_GR and not return_assym: return mean,gauss_err,GR
#   elif not return_GR and return_assym: return mean,gauss_err,neg_err,pos_err
#   elif return_GR and return_assym: return mean,gauss_err,GR,neg_err,pos_err
#   else: return mean,gauss_err
  
###############################################################################
def GetSamples(conv_length,N,n_chains=None,chain_filenames=None):
  """
  Get N samples from chains
  
  """
  
  #get chain names tuple
  if n_chains == None and chain_filenames == None:
    chain_filenames=("MCMC_chain",)  
  if n_chains != None:
    chain_filenames = ["MCMC_chain_%d" % i for i in range(1,n_chains+1)]
  if type(chain_filenames) is str:
    chain_filenames = [chain_filenames,]
    
  #merge files into large matrix
  X = np.load(chain_filenames[0]+'.npy')[conv_length:] #read in data file  
  for i in range(1,len(chain_filenames)):
    X = np.concatenate((X,np.load(chain_filenames[i]+'.npy')[conv_length:]))
  
  #create random index
  Q = X.shape[0] #length of array
  index = np.random.randint(0,Q,N) #get N random integers
  
  #return the random samples:
  return X[:,1:][index]
  
###############################################################################

def GetBestFit(n_chains=None,chain_filenames=None,conv_length=0):
  """
  Extract the maximum likelihood/posterior parameters from the chain files.
  """
  
  #get chain names tuple
  if n_chains == None and chain_filenames == None:
    chain_filenames=("MCMC_chain",)  
  if n_chains != None:
    chain_filenames = ["MCMC_chain_%d" % i for i in range(1,n_chains+1)]
  if type(chain_filenames) is str:
    chain_filenames = [chain_filenames,]
  
  #get number of parmaeters
  #no_pars = len(open(chain_filenames[0],'r').readline().split())-1
  no_pars = np.load(chain_filenames[0]+'.npy')[0].size-1
  X = np.load(chain_filenames[0]+'.npy')[conv_length:] #read in data file
  
  for i in range(1,len(chain_filenames)):
    X = np.concatenate((X,np.load(chain_filenames[i]+'.npy')[conv_length:]))
  
  index = np.where(X[:,0] == np.nanmax(X[:,0])) #get the index of maximum posterior prob
  
#  print index
#  print "Likelihood:", X[index][0][0]
  return X[index][0][1:] #return best fit parameters

###############################################################################

def PlotChains(conv_length,p=None,n_chains=None,chain_filenames=None,saveplot=False,filename="ChainPlot.pdf",labels=None):
  """
  Plot the MCMC chains. Can be very slow for long chains, should probably add a range.
  
  """
  
  #get chain names tuple
  if n_chains == None and chain_filenames == None:
    chain_filenames=("MCMC_chain",)  
  if n_chains != None:
    chain_filenames = ["MCMC_chain_%d" % i for i in range(1,n_chains+1)]
  if type(chain_filenames) is str:
    chain_filenames = [chain_filenames,]
  
  if p==None: #get total number of parmaeters if not supplued
    #no_pars = len(open(chain_filenames[0],'r').readline().split())-1
    no_pars = np.load(chain_filenames[0]+'.npy')[0].size-1
    p=range(no_pars)
  else:
    no_pars = len(p)
  
  if labels == None: #create labels for plots if not provided
    labels = ['p[%d]' % p[q] for q in range(no_pars)]
  
#  pylab.figure(len(pylab.get_fignums())+1,(9,no_pars*2)) #new figure
#  pylab.clf() #clear figure
    
  for i,file in enumerate(chain_filenames):

    #loop over the parameters
    for q in range(no_pars):
      
      Data = np.load(file+'.npy')[:,p[q]] #read in data - data is stored in q+1
      
      #create axes 1 and plot chain
      #axes [x(left),y(lower),width,height]
      pylab.axes([0.1,1.-(q+1-0.15)*(1./no_pars),0.6,(1./no_pars)*0.75])
      pylab.plot(Data,'-')
      pylab.axvline(conv_length,color='r',linestyle='--')
      pylab.ylabel(labels[q])
      if (q+1) == no_pars: pylab.xlabel("N")

      #create axes 2 and plot histogram
      pylab.axes([0.75,1.-(q+1-0.15)*(1./no_pars),0.2,(1./no_pars)*0.75])
      pylab.hist(Data[conv_length:],20,histtype='step')
      pylab.xticks([])
      pylab.yticks([])

  if saveplot: #save the plots
    print ("Saving chain plots...")
    pylab.savefig(filename)
  
###############################################################################

def PlotCorrelations(conv_length,p=None,n_chains=None,chain_filenames=None,saveplot=False,filename="CorrelationPlot.pdf",labels=None,n_samples=500):
  """
  Make correlation plots from MCMC output, plus histograms of each of the parameters.
  
  """
  
  #get chain names tuple
  if n_chains is None and chain_filenames is None:
    chain_filenames=("MCMC_chain",)  
  if n_chains is not None:
    chain_filenames = ["MCMC_chain_%d" % i for i in range(1,n_chains+1)]
  if type(chain_filenames) is str:
    chain_filenames = [chain_filenames,]
  
  if p is None: #get total number of parmaeters if not supplied
#    no_pars = len(open(chain_filenames[0],'r').readline().split())-1
    no_pars = np.load(chain_filenames[0]+'.npy')[0].size-1
    p=range(no_pars)
  else:
    no_pars = len(p)

  if labels is None: #create labels for plots if not provided
    labels = ['p[%d]' % p[q] for q in range(no_pars)]

  #new fig and adjust plot params
#  pylab.figure(len(pylab.get_fignums())+1,(no_pars*2,no_pars*2))  
  pylab.clf() #clear figure
#  fig = pylab.figure(len(pylab.get_fignums()),(10,10))
  pylab.rcParams['lines.markersize'] = 2.
  pylab.rcParams['font.size'] = 10
  pylab.subplots_adjust(left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.00001,hspace=0.00001)
  
  for file in chain_filenames:  
    Data = np.load(file+'.npy')
    index = np.random.randint(conv_length,Data[:,0].size,n_samples) #randomly sample the data for plots

    for i in range(no_pars): #loop over the parameter indexes supplied
      for q in range(i+1):
        pylab.subplot(no_pars,no_pars,i*no_pars+q+1,xticks=[],yticks=[]) #select subplot
        
        if(i==q):
          hdata = pylab.hist(Data[:,p[i]+1][conv_length:],20,histtype='step',normed=1)   
          pylab.xlim(hdata[1].min(),hdata[1].max())
          pylab.ylim(0,hdata[0].max()*1.1)
        else: pylab.plot(Data[:,p[q]+1][index],Data[:,p[i]+1][index],'.')
        
        if q == 0: pylab.ylabel(labels[i])
        if i == (no_pars-1): pylab.xlabel(labels[q])
  
  if saveplot: #save the plots
    if type(filename) == list or type(filename) == tuple:
      for name in filename:
        print ("Saving correlation plot...")
        pylab.savefig(name,dpi=300)    
    else:
      print ("Saving correlation plot...")
      pylab.savefig(filename,dpi=300)

###############################################################################

def PlotCorrelations_inv(conv_length,p=None,n_chains=None,chain_filenames=None,saveplot=False,filename="CorrelationPlot.pdf",labels=None,n_samples=500):
  """
  Plot correlations in upper right corner.
  
  """
  
  #get chain names tuple
  if n_chains == None and chain_filenames == None:
    chain_filenames=("MCMC_chain",)  
  if n_chains != None:
    chain_filenames = ["MCMC_chain_%d" % i for i in range(1,n_chains+1)]
  
  if p==None: #get total number of parmaeters if not supplied
    #no_pars = len(open(chain_filenames[0],'r').readline().split())-1
    no_pars = np.load(chain_filenames[0]+'.npy')[0].size-1
    p=range(no_pars)
  else:
    no_pars = len(p)

  if labels == None: #create labels for plots if not provided
    labels = ['p[%d]' % p[q] for q in range(no_pars)]

  #new fig and adjust plot params
#  pylab.figure(len(pylab.get_fignums())+1,(no_pars*2,no_pars*2))  
  pylab.clf() #clear figure
#  fig = pylab.figure(len(pylab.get_fignums()),(10,10))
  pylab.rcParams['lines.markersize'] = 2.
  pylab.rcParams['font.size'] = 10
  pylab.subplots_adjust(left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.00001,hspace=0.00001)
  
  for file in chain_filenames:  
    Data = np.load(file+'.npy')
    index = np.random.randint(conv_length,Data[:,0].size,n_samples) #randomly sample the data for plots

    for i in range(no_pars): #loop over the parameter indexes supplied
      for q in range(i+1):
        pylab.subplot(no_pars,no_pars,(no_pars-i)*no_pars-q,xticks=[],yticks=[]) #select subplot
        
        print (i,q,no_pars,no_pars,(i+1)*no_pars-q)
        
        if(i==q):
          hdata = pylab.hist(Data[:,p[i]+1][conv_length:],20,histtype='step',normed=1)   
          pylab.xlim(hdata[1].min(),hdata[1].max())
          pylab.ylim(0,hdata[0].max()*1.1)
        else: pylab.plot(Data[:,p[q]+1][index],Data[:,p[i]+1][index],'.')
        
        #place axis at top and right
        pylab.gca().yaxis.set_label_position('right')
        pylab.gca().xaxis.set_label_position('top')
        
        if q == 0: pylab.ylabel(labels[i])
        if i == (no_pars-1): pylab.xlabel(labels[q])

  if saveplot: #save the plots
    if type(filename) == list or type(filename) == tuple:
      for name in filename:
        print ("Saving correlation plot...")
        pylab.savefig(name,dpi=300)    
    else:
      print ("Saving correlation plot...")
      pylab.savefig(filename,dpi=300)

###############################################################################

def GelRub(chain_files,col,l):
  """
  Compute the Gelman and Rubin statistic and errors for one parameter (in column col).
  """
  
  no_chains = len(chain_files)
  L = int(len(np.load(chain_files[0]+'.npy')[:,0])-l)
  
  #create empty arrays for data
  data = np.empty(no_chains*L,dtype=np.float64).reshape(no_chains,L)
  mean = np.empty(no_chains,dtype=np.float64)
  var = np.empty(no_chains,dtype=np.float64)
  
  #get means and variances for each chain
  for i in range(no_chains):
#    data[i] = np.loadtxt(chain_files[i],usecols=(col,))[l:]
    data[i] = np.load(chain_files[i]+'.npy')[:,col][l:]
    mean[i] = data[i].mean(dtype=np.float64)
    var[i] = data[i].var(dtype=np.float64)
  
  #test if data is fixed (non trivial because of rounding errors affecting mean/var)
  if np.all(data.flatten() == data.flatten()[::-1]): #test array is the same as its reverse
    return data[0,0],data[0,0],0.,0.,0.,-1
  
  if no_chains > 1: #calculate GR statistic
    W = var.mean(dtype=np.float64) #mean of the variances
    B = mean.var(dtype=np.float64) #variance of the means
    GR = np.sqrt((((L-1.)/float(L))*W + B) / W) #GR stat
  else:
    GR = -2
    
  #mean
  dist_mean = data.mean(dtype=np.float64)
  
  #median
  dist_median = np.median(data)
  
  #stddev error
  err = data.std(dtype=np.float64)
  
  #get 1sig errors as range to 0.159, 1-0.159
  data = data.flatten()
  data.sort()
  lower = data[int(0.159*data.size)]
  upper = data[int((1.-0.159)*data.size)]
  
  #return best fit, gaussian error, asym_error (+,-), GR stat
  return dist_mean,dist_median,err,upper-dist_median,dist_median-lower,GR

###############################################################################

def PlotCorrelations_im(conv_length,p=None,n_chains=None,chain_filenames=None,saveplot=False,filename="CorrelationPlot.pdf",labels=None,n_samples=500,cmap=None,\
  left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.0001,hspace=0.0001):

  #get chain names tuple
  if n_chains is None and chain_filenames is None:
    chain_filenames=("MCMC_chain",)  
  if n_chains is not None:
    chain_filenames = ["MCMC_chain_%d" % i for i in range(1,n_chains+1)]
  if type(chain_filenames) is str:
    chain_filenames = [chain_filenames,]
  
  print (chain_filenames)
  
  if p==None: #get total number of parmaeters if not supplied
    no_pars = np.load(chain_filenames[0])[0].size-1
    p=range(no_pars)
  else:
    no_pars = len(p)

  if labels == None: #create labels for plots if not provided
    labels = []
    for q in range(no_pars):
      labels.append('p[%d]' % p[q]) 
  
  print (no_pars)
  pylab.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)
  
#   Data = [0,0,0,0]
#   index = [0,0,0,0]
#   for s,file in enumerate(chain_filenames):  
#     Data[s] = np.loadtxt(file)
#     index[s] = np.random.randint(conv_length,Data[s][:,0].size,n_samples) #randomly sample the data for plots
  
  Data = [np.load(file) for file in chain_filenames]
  index = [np.random.randint(conv_length,Data[s][:,0].size,n_samples) for s in range(len(chain_filenames))]
  
  for i in range(no_pars): #loop over the parameter indexes supplied
    for q in range(i+1):
      pylab.subplot(no_pars,no_pars,i*no_pars+q+1,xticks=[],yticks=[]) #select subplot
      
      if(i==q):
        hdata = [pylab.hist(d[:,p[i]+1][conv_length:],20,histtype='step',normed=1,color='k') for d in Data]
        pylab.xlim(hdata[0][1].min(),hdata[0][1].max())
        pylab.ylim(0,hdata[0][0].max()*1.1)
        #pylab.axis('off')
      else:
        #pylab.plot(Data[:,p[q]+1][index],Data[:,p[i]+1][index],'.',ms=3)
        temp_dat1 = np.concatenate([d[:,p[q]+1][conv_length:] for d in Data])
#        temp_dat1 = np.concatenate([Data[0][:,p[q]+1][conv_length:],Data[1][:,p[q]+1][conv_length:],Data[2][:,p[q]+1][conv_length:],Data[3][:,p[q]+1][conv_length:]])
        temp_dat2 = np.concatenate([d[:,p[i]+1][conv_length:] for d in Data])
#        temp_dat2 = np.concatenate([Data[0][:,p[i]+1][conv_length:],Data[1][:,p[i]+1][conv_length:],Data[2][:,p[i]+1][conv_length:],Data[3][:,p[i]+1][conv_length:]])
        DensityPlot(temp_dat1,temp_dat2,bins=(20,20),plot=True,plot_contour=1,cmap=cmap,interpolation=None)
      if q == 0: pylab.ylabel(labels[i])
      if i == (no_pars-1): pylab.xlabel(labels[q])

##########################################################################################
# def PlotCorrelations_inv_im(conv_length,p=None,n_chains=None,chain_filenames=None,saveplot=False,filename="CorrelationPlot.pdf",labels=None,n_samples=500,cmap=None):
# 
#   #get chain names tuple
#   if n_chains == None and chain_filenames == None:
#     chain_filenames=("MCMC_chain",)  
#   if n_chains != None:
#     chain_filenames = []
#     for i in range(n_chains):
#       chain_filenames.append("MCMC_chain_%d" % (i+1))
#   
#   if p==None: #get total number of parmaeters if not supplied
#     no_pars = len(open(chain_filenames[0],'r').readline().split())-1
#     p=range(no_pars)
#   else:
#     no_pars = len(p)
# 
#   if labels == None: #create labels for plots if not provided
#     labels = []
#     for q in range(no_pars):
#       labels.append('p[%d]' % p[q]) 
#   
#   Data = [0,0,0,0]
#   index = [0,0,0,0]
#   for s,file in enumerate(chain_filenames):  
#     Data[s] = np.loadtxt(file)
#     index[s] = np.random.randint(conv_length,Data[s][:,0].size,n_samples) #randomly sample the data for plots
# 
#   for i in range(no_pars): #loop over the parameter indexes supplied
#     for q in range(i+1):
#       pylab.subplot(10,10,(no_pars-i)*10-q,xticks=[],yticks=[]) #select subplot
#               
#       if(i==q):
#         hdata = pylab.hist(Data[0][:,p[i]+1][conv_length:],20,histtype='step',normed=1,color='k')   
#         hdata = pylab.hist(Data[1][:,p[i]+1][conv_length:],20,histtype='step',normed=1,color='r')   
#         hdata = pylab.hist(Data[2][:,p[i]+1][conv_length:],20,histtype='step',normed=1,color='b')   
#         hdata = pylab.hist(Data[3][:,p[i]+1][conv_length:],20,histtype='step',normed=1,color='g')   
#         pylab.xlim(hdata[1].min(),hdata[1].max())
#         pylab.ylim(hdata[0].min(),hdata[0].max()*1.1)
#       else:
#         #pylab.plot(Data[:,p[q]+1][index],Data[:,p[i]+1][index],'.',ms=3)
#         temp_dat1 = np.concatenate([Data[0][:,p[q]+1][conv_length:],Data[1][:,p[q]+1][conv_length:],Data[2][:,p[q]+1][conv_length:],Data[3][:,p[q]+1][conv_length:]])
#         temp_dat2 = np.concatenate([Data[0][:,p[i]+1][conv_length:],Data[1][:,p[i]+1][conv_length:],Data[2][:,p[i]+1][conv_length:],Data[3][:,p[i]+1][conv_length:]])
#         DensityPlot(temp_dat1,temp_dat2,bins=(40,40),plot=True,plot_contour=1,cmap=cmap,interpolation=None)
#       #place axis at top and right
#       pylab.gca().yaxis.set_label_position('right')
#       pylab.gca().xaxis.set_label_position('top')
#       
#       if q == 0: pylab.ylabel(labels[i])
#       if i == (no_pars-1): pylab.xlabel(labels[q])

##########################################################################################
def DensityPlot(x,y,bins=(50,50),range=None,sigma=1.5,plot=False,interpolation='nearest',cmap=None,plot_contour=0):
  
  #get the 2d histogram
  H,a,b = np.histogram2d(y,x,bins=bins,range=range)
  H /= H.max()
  
  #smooth using a Gaussian filter
  if sigma: SH = ndimage.gaussian_filter(H, sigma=sigma, order=0)
  else: SH = H[:]
  SH = (SH - SH.min())/(SH.max()-SH.min())

  q = SH.flatten()
  q.sort()
  qsum = np.cumsum(q)
  qsum /= qsum.max()
  s3 = q[np.where(qsum<0.0027000000000000357)[0][-1]]
  s2 = q[np.where(qsum<0.045499999999999985)[0][-1]]
  s1 = q[np.where(qsum<0.31730000000000003)[0][-1]]
  
  if range==None:
    extent = range
  else: extent = [range[0][0],range[0][1],range[1][0],range[1][1]]
  
  extent = None
  
  if plot:
    pylab.imshow(np.sqrt(SH),extent=extent,origin='lower',interpolation=interpolation,cmap=cmap,vmin=0.0,vmax=1.,rasterized=1)
  if plot_contour:
    pylab.contour(SH,extent=extent,origin='lower',levels=(s2,s1),colors='k')
#    pylab.contour(SH,extent=extent,origin='lower',levels=(s3,),colors='k')#,linestyles='dashed')
  #  pylab.xlim(b.min(),b.max())
  #  pylab.ylim(a.min(),a.max())
  
  #return the image
  return SH






