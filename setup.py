from setuptools import setup, Extension
import numpy as np

setup(

  name = "Infer", version = "1.0",
  description='Python version of MCMC, plus other inference codes under development', 
  author='Neale Gibson',
  author_email='n.gibson@tcd.ie',

  packages=['Infer'],
  package_dir={'Infer':'src'},
  
  #and extension package for solving toeplitz matrices...
  ext_modules = [
    Extension("Infer.LevinsonTrenchZoharSolve",sources=["src/LevinsonTrenchZoharSolve.c"],),
    ],

  include_dirs=[np.get_include(),],      

  )
