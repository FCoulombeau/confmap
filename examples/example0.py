# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 05:05:15 2019

@author: FC
"""

import confmap as cm
import numpy as np

im = cm.HyperbolicTiling('./Reflets.jpg',0,'',600,600)

im.transform(sommets=(np.inf,6,4,6),nbit=20,backcolor=[255,255,255], delta=1e-3)