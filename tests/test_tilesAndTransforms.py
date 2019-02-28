# -*- coding: utf-8 -*-
"""Image transformation test meant to be run with pytest."""
import sys

import pytest

from confmap import ImageTransform
from confmap import HyperbolicTiling

sys.path.append("tests")

def test_tilesAndTransform():
    
    im=ImageTransform('./examples/sample1.png',0,data=None
                ,c=1.*(1.+0.j),r=1.*(1.+0.j)
                ,d=0.08+0.55j,output_width=750
                ,output_height=1000,blur=False,smoothshift=-0,shift=0.)
    im.mirror(Y=2,X=1)
    res=im.transform(print_and_save=False)
    HT=HyperbolicTiling('./examples/sample1.png',prefix='./examples/',suffix='0',
                        output_width=1550,output_height=640,data=res)
    im=ImageTransform(HT,d=0.04)
    im.arctan()
    im.similitude(c=1.9)
    HT.transform(c=0.95,d=0.+0.0j,backcolor=True,vanishes=False,
                nbit=25,delta=0e-3,print_and_save=True,
                sommets=(6,4,4,4,6,4,4,4))
    return True

if __name__ == "__main__":
    pytest.main()