"""
Updated correlation plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Arrow
import scipy.ndimage as ndimage

###############################################################################

def CorrelationAxes(N,inv=False,labels=None,left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.03,hspace=0.03):
  """
  Returns axes for correlation plots
  """
  
  plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)
  
  ax = {}
  labels = [r'$p_{}$'.format(i+1) for i in range(N)]
  
  #create normal axes
  if not inv:
    for i in range(N): #loop over the parameter indexes supplied
      for q in range(i+1):
        ax['{}{}'.format(i,q)] = plt.subplot(N,N,i*N+q+1,xticks=[],yticks=[])
        #add labels...
        if i == (N-1): ax['{}{}'.format(i,q)].set_xlabel(labels[q])
      ax['{}{}'.format(i,0)].set_ylabel(labels[i])
        
  #or inverse axes
  else:
    for i in range(N):
      for q in range(i+1):
         ax['{}{}'.format(i,q)] = plt.subplot(N,N,(N-i)*N-q,xticks=[],yticks=[])

  return ax
  
###############################################################################

def CorrelationHist(X,ax=False,inv=False,**kwargs):
  
  #get no of dimensions
  N = X.shape[1]
  
  #create axes if not provided
  if ax==False:  
    ax = CorrelationAxes(N,inv=inv)
  
  #make a plot of the histograms accross the diagonal
  for i in range(N):
    ax['{}{}'.format(i,i)].hist(X[:,i],20,histtype='step',normed=1,**kwargs)

###############################################################################

def CorrelationScatterPlot(X,ax=False,samples=100,inv=False,alpha=0.6,zorder=3,**kwargs):
  
  #get no of dimensions
  S,N = X.shape
  ind = np.random.randint(0,S,samples)
  
  #create axes if not provided
  if ax==False:  
    ax = CorrelationAxes(N,inv=inv)

  #loop over the axes (except-diagonals) and make scatter plot
  for i in range(N): #loop over the parameter indexes supplied
    for q in range(i):
      ax['{}{}'.format(i,q)].plot(X[:,q][ind],X[:,i][ind],'.',alpha=alpha,zorder=zorder,**kwargs)
  
###############################################################################

def CorrelationEllipses(X=None,mu=None,K=None,ax=False,samples=100,inv=False,alpha=0.6,zorder=3,**kwargs):
    
  #get covariance matrix of X if not given
  if K is None:
    if X is None:
      raise ValueError, "must provide either X or K!"
    #get no of dimensions
    N = X.shape[1]
    #get mean and covariance
    K = np.cov(X.T)
    mu = X.mean(axis=0)
    #check if parameter is fixed
    for i in range(N):
      if np.all(X[:,i] == X[:,i][::-1]): #check if array equals its reverse
        K[i,i] = 0.
  
  else: #use K
    N = K.shape[0]
    if mu is None:
      mu = np.zeros(N)
  
  #create axes if not provided
  if ax==False:  
    ax = CorrelationAxes(N,inv=inv)
  
  #loop over the axes (except-diagonals) and make scatter plot
  for i in range(N): #loop over the parameter indexes supplied
    for q in range(i):

      if K[i,i] == 0. or K[q,q] == 0.:
        continue
      #first get 2D covariance and mean:
      m = [mu[q],mu[i]]      
      K_t = np.diag([K[q][q],K[i][i]]) #note that the axes are swapped - q is the x-axis!
      K_t[1,0],K_t[0,1] = K[i][q],K[i][q]
      
      #get eigen decomposition
      w,v = np.linalg.eig(K_t) #first get eigen decomposition
      #define ellipse for 1,2 and 3 sigma
      angle = np.arctan(v[:,0][1]/v[:,0][0]) * 180./np.pi #get angle from principle component

      e = [Ellipse(m,2*np.sqrt(w[0])*np.sqrt(n),2*np.sqrt(w[1])*np.sqrt(n),\
          angle,lw=1,fill=True,alpha=0.2,ec='k',zorder=2) for n in [2.295817,6.1801,11.83]]

      for n in range(3): ax['{}{}'.format(i,q)].add_patch(e[n])
      ax['{}{}'.format(i,q)].plot()
      
###############################################################################

def CorrelationDensity(X,ax=False,inv=False,Nz=5,Nm=5,Ng=2,alpha=0.6,zorder=3,**kwargs):
  
  #get no of dimensions
  S,N = X.shape
  
  #create axes if not provided
  if ax==False:  
    ax = CorrelationAxes(N,inv=inv)
  
  #loop over the axes (except-diagonals) and make scatter plot
  for i in range(N): #loop over the parameter indexes supplied
    for q in range(i):
      
      #set means and ranges for both axes
      mq,sq = X[:,q][:].mean(),X[:,q][:].std()
      mi,si = X[:,i][:].mean(),X[:,i][:].std()
      rq = np.linspace(mq-10.*sq,mq+10.*sq,50)
      ri = np.linspace(mi-10.*si,mi+10.*si,50)
      
      if np.isclose(sq,0.0) or np.isclose(si,0.0):
        continue
      
      #get 2D histogram
      H,a,b = np.histogram2d(X[:,q][:],X[:,i][:],bins=(rq,ri),normed=1)
      a_mid = a[:-1] + (a[1]-a[0])/2. #convert to midpoints (hist returns limits)
      b_mid = b[:-1] + (b[1]-b[0])/2.
      
      #find the values for the contours via cumulative distribution
      H = H[:] / H.max() #normalise the histogram
      fl = H.flatten() #flatten
      fl.sort() #and sort
      qsum = np.cumsum(fl) #get cumulative distribution
      qsum /= qsum.max() #and normalise
      #finally, get closest values in array to 1,2,3 sigma limits
      ind1 = np.abs(qsum - 0.3173).argmin()
      ind2 = np.abs(qsum - 0.0455).argmin()
      ind3 = np.abs(qsum - 0.0027).argmin()
      s1,s2,s3 = fl[ind1],fl[ind2],fl[ind3]
      
      #smooth image for contour plot
      a_mid = ndimage.zoom(a_mid,Nz)
      b_mid = ndimage.zoom(b_mid,Nz)
      H = ndimage.zoom(H,Nz)
      #apply median filter and gaussian filter to smooth
      H = ndimage.median_filter(H, size=Nm*Nz)
      H = ndimage.gaussian_filter(H, sigma=Ng)
      
      #only plot a subset of the contour plot
      filt = np.where( (a_mid > mq-5.*sq) * (a_mid < mq+5.*sq) )
      
#      return a_mid,b_mid,H,filt
      ax['{}{}'.format(i,q)].contour(a_mid[filt],b_mid[filt],H[filt].T[filt],origin='lower',levels=(s3,s2,s1),colors='r',zorder=30)

###############################################################################

def CorrelationPlot(conv_length,p=None,inv=False,n_chains=None,chain_filenames=None,saveplot=False,filename="CorrelationPlot.pdf",labels=None,n_samples=500):
  """
  Make correlation plots from MCMC output, plus histograms of each of the parameters.
  
  """
  
  #get chain names tuple
  if n_chains is None and chain_filenames is None:
    chain_filenames=("MCMC_chain",)  
  if n_chains is not None:
    chain_filenames = ["MCMC_chain_%d" % i for i in range(1,n_chains+1)]
  if n_chains is None:
    n_chains = len(chain_filenames)
  
  X_temp = np.load(chain_filenames[0]+'.npy')
  ch_length = X_temp[:,0].size - conv_length

  if p is None: #get total number of parmaeters if not supplied
    no_pars = X_temp[0].size-1
    p=range(no_pars)
  else:
    no_pars = len(p)

  if labels is None: #create labels for plots if not provided
    labels = ['p[%d]' % p[q] for q in range(no_pars)]

  #get axes
  ax = CorrelationAxes(no_pars,inv=inv)
  
  #create array to store data
  X = np.empty((n_chains,ch_length,no_pars))
  
  #get data
  for i,file in enumerate(chain_filenames):  
    X[i] = np.load(file+'.npy')[conv_length:,1:]
  
  #remove constant values?
  #
  #
  #
  
#  X -= X.mean(axis=1)
    
  for i in range(n_chains):
    CorrelationScatterPlot(X[i],ax=ax,samples=n_samples)

  for i in range(n_chains):
    CorrelationHist(X[i],ax=ax)
  
  #X[:,0] -= X[:,0].mean()
  
  #plot histograms and density
#  CorrelationHist(X,ax=ax)
#  CorrelationDensity(X[i],ax=ax)
  CorrelationDensity(X.reshape(-1,no_pars),ax=ax)
  
  print (X.shape)
  print (ax.keys())
  return X
  
    
  
  
    
  
#   if saveplot: #save the plots
#     if type(filename) == list or type(filename) == tuple:
#       for name in filename:
#         print "Saving correlation plot..."
#         pylab.savefig(name,dpi=300)    
#     else:
#       print "Saving correlation plot..."
#       pylab.savefig(filename,dpi=300)





    