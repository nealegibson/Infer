#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse,Arrow


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
    pylab.imshow(np.sqrt(SH),extent=extent,origin='lower',interpolation=interpolation,cmap=cmap,vmin=0.0,vmax=1.)
  if plot_contour:
    pylab.contour(SH,extent=extent,origin='lower',levels=(s2,s1),colors='k')
#    pylab.contour(SH,extent=extent,origin='lower',levels=(s3,),colors='k')#,linestyles='dashed')
  #  pylab.xlim(b.min(),b.max())
  #  pylab.ylim(a.min(),a.max())
  
  #return the image
  return SH

plt.figure(1)

#or draw from a multivariate distribution
mu = [1,1]
K = [[1,1,],[1,2]]
X = np.random.multivariate_normal(mu,K,1000)

#get covariance matrix
K = np.cov(X.T)
w,v = np.linalg.eig(K) #either use the eigenvalues directly
u,w,v = np.linalg.svd(K) #or use single value decomposition
angle = np.arctan(v[:,0][1]/v[:,0][0]) * 180./np.pi #get angle from principle component

#create arrows for eigenvectors
a = [Arrow(X[:,0].mean(),X[:,1].mean(),v[:,i][0]*np.sqrt(w[i]),v[:,i][1]*np.sqrt(w[i]),width=0.2,zorder=10,color='k') for i in range(2)]

#create ellipses for 3 sigma
e = [Ellipse([X[:,0].mean(),X[:,1].mean()],np.sqrt(w[0])*n*2,np.sqrt(w[1])*n*2,angle,lw=1,fill=True,alpha=0.2,ec='k',zorder=6) for n in [1,2,3]]

#add the patches and make the scatter plot
ax = plt.axes()
[ax.add_patch(a[i]) for i in range(2)]
[ax.add_patch(e[n]) for n in range(3)]
ax.scatter(X[:,0],X[:,1],alpha=0.5,zorder=5,color='r')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_aspect('equal','datalim')

#
# multidimensional case
##

plt.figure(2)

mu = [1,1,3]
K = np.array([[1,0,0],[0,1,1],[0,1,2]])
X = np.random.multivariate_normal(mu,K,100000)
N = len(np.diag(K))

plt.subplots_adjust(left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.03,hspace=0.03)
n_samples = 500
ind = np.random.randint(0,X[:,0].size,n_samples)

#create axes array
ax = {}
for i in range(len(np.diag(K))): #loop over the parameter indexes supplied
  for q in range(i+1):
    ax['{}{}'.format(i,q)] = plt.subplot(N,N,i*N+q+1,xticks=[],yticks=[])
    
#plot error ellipses:
for i in range(len(np.diag(K))): #loop over the parameter indexes supplied
  for q in range(i):
    
    #define ellipses for 3 sigma
    #first con
    K_t = np.diag([K[q][q],K[i][i]]) #note that the axes are swapped - q is the x-axis!
    K_t[1,0],K_t[0,1] = K[i][q],K[i][q]
    
    #K_t = K.compress([1,1,0],axis=0).compress([1,1,0],axis=1)
    w,v = np.linalg.eig(K_t) #first get eigen decomposition
    angle = np.arctan(v[:,0][1]/v[:,0][0]) * 180./np.pi #get angle from principle component
    e = [Ellipse([X[:,q].mean(),X[:,i].mean()],np.sqrt(w[0])*n*2,np.sqrt(w[1])*n*2,angle,lw=1,fill=True,alpha=0.2,ec='k',zorder=6) for n in [1,2,3]]
    #e = [Ellipse([X[:,q].mean(),X[:,i].mean()],np.sqrt(w[q])*n*2,np.sqrt(w[i])*n*2,angle,lw=1,fill=True,alpha=0.2,ec='k',zorder=6) for n in [1,2,3]]
    
    for n in range(3): ax['{}{}'.format(i,q)].add_patch(e[n])
    plt.draw()
    plt.plot()

#create plots in the axes
for i in range(len(np.diag(K))): #loop over the parameter indexes supplied
  ax['{}{}'.format(i,i)].hist(X[:,i],20,histtype='step',normed=1)

#create scatter plots
for i in range(len(np.diag(K))): #loop over the parameter indexes supplied
  for q in range(i):
#    ax['{}{}'.format(i,q)].plot(X[:,q][ind],X[:,i][ind],'.')
#    ax['{}{}'.format(i,q)].hexbin(X[:,q][ind],X[:,i][ind],gridsize=30,alpha=0.2)
    ax['{}{}'.format(i,q)].scatter(X[:,q][ind],X[:,i][ind],alpha=0.2)

#create density plots
for i in range(len(np.diag(K))): #loop over the parameter indexes supplied
  for q in range(i):
#    ax['{}{}'.format(i,q)].plot(X[:,q][ind],X[:,i][ind],'.')
#    ax['{}{}'.format(i,q)].hexbin(X[:,q][ind],X[:,i][ind],gridsize=30,alpha=0.2)
    
    H,a,b = np.histogram2d(X[:,q][:],X[:,i][:],bins=(20,20),normed=1)
    a_mid = a[:-1] + (a[1]-a[0])/2.
    b_mid = b[:-1] + (b[1]-b[0])/2.

    #smooth using a Gaussian filter
    #if sigma: SH = ndimage.gaussian_filter(H, sigma=sigma, order=0)
    #else: SH = H[:]
    SH = H[:]
    SH = (SH - SH.min())/(SH.max()-SH.min())

    fl = SH.flatten()
    fl.sort()
    qsum = np.cumsum(fl)
    qsum /= qsum.max()
    s3 = fl[np.where(qsum<0.0027000000000000357)[0][-1]]
    s2 = fl[np.where(qsum<0.045499999999999985)[0][-1]]
    s1 = fl[np.where(qsum<0.31730000000000003)[0][-1]]

    ax['{}{}'.format(i,q)].hist2d(X[:,q][:],X[:,i][:],(50,50),alpha=1,zorder=1,cmap='Reds')
    ax['{}{}'.format(i,q)].contour(a_mid,b_mid,SH,origin='lower',levels=(s3,s2,s1),colors='k')
