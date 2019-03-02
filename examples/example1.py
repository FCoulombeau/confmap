# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 05:05:15 2019

@author: FC
"""

import confmap as cm
import numpy as np

im = cm.HyperbolicTiling('./Reflets.jpg',1,'',600,600)

im.transform(sommets=(6,4),nbit=20,backcolor=[255,255,255], delta=1e-3)