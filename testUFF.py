# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 01:43:47 2018

@author: kb-
Test writing time-series dataset 58b and 58 files
"""

import numpy as np
import pickle
import UFF

mask = pickle.load(open('mask.pkl','rb'))#timeseries dataset 58 file information

fs = 51200.0
dt = np.dtype(np.single)
t=np.arange(0,10,1/fs).astype(dt)

A=[1,2,3]
f=[500,8000,10]

for i in range(3):
    y=A[i]*np.sin(2*np.pi*f[i]*t)
    #write dataset 58b file
    UFF.write(file='test58b.uff',mask=mask,fs=fs,binary=1,data=y,point=str(i),ordinate_axis_units_lab='g')
    
    #write dataset 58 file
    UFF.write(file='test58.uff',mask=mask,fs=fs,binary=0,data=y,point=str(i),ordinate_axis_units_lab='g')