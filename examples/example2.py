# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 05:05:15 2019

@author: FC
"""

import confmap as cm
import confmap.hyperbolic as hyp
import numpy as np
import matplotlib.pyplot as plt

a=0.39797542678479064
b=hyp._reflexion(-0.32196888361251824+0.23392408663890288j,-0.32196888361251824-0.23392408663890288j)(a)
d=(np.arctanh(a)-np.arctanh(b))/np.pi
thet=np.pi/2-np.arcsin(1/d/3)/2

im = cm.HyperbolicTiling('./Reflets.jpg',2,'',800,450)

tr=cm.ImageTransform(im,c=np.exp(-1j)*0.8,d=-0.1j-0.07)
tr.arctan(hyperb=True,dec=np.pi/np.tan(thet))
#tr.similitude(auto=False,c=np.exp(1j*thet)*np.cos(thet))
tr.exp(auto=False,angle=False,c=2*np.pi*np.exp(1j*thet)*np.cos(thet))
im.rotation(0,np.pi/10)
im.translation(0.2j)
im.translation(-0.2j,pre=False)
im.rotation(0,-np.pi/10,pre=False)
im.translation(c=-a,pre=False)

im2=im.transform(c=0.9,sommets=(5,4),nbit=20,backcolor=[255,255,255], delta=0e-3)
ind=(im2==[255,255,255])


im = cm.HyperbolicTiling('./Reflets.jpg',3,'',800,450)

tr=cm.ImageTransform(im,c=np.exp(-1j)*0.8,d=-0.1j-0.07)
tr.arctan(hyperb=True,dec=np.pi/np.tan(thet))

#tr.similitude(auto=False,c=np.exp(1j*thet)*np.cos(thet),d=(1/np.cos(thet)-1/np.exp(1j*thet))/np.sin(thet))
tr.exp(auto=False,angle=False,c=2*np.pi*np.exp(1j*thet)*np.cos(thet),d=(1/np.cos(thet)-1/np.exp(1j*thet))/np.sin(thet))
im.rotation(0,np.pi/10)
im.translation(0.2j)
im.translation(-0.2j,pre=False)
im.rotation(0,-np.pi/10,pre=False)
im.translation(c=-a,pre=False)

im2[ind]=im.transform(c=0.9,sommets=(5,4),nbit=20,backcolor=[255,255,255], delta=0e-3)[ind]
plt.imsave('./Reflets-4.jpg',im2)