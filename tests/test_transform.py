# -*- coding: utf-8 -*-
"""Image transformation test meant to be run with pytest."""
import sys

import pytest

from confmap import ImageTransform

sys.path.append("tests")

def test_transform():
    im=ImageTransform('./examples/sample1.png',2,prefix='./examples/',data=None,
                       r=0.01,c=1.,d=0.-0.j,
                       output_height=900,output_width=1600,
                       blur=True,shift=0,smoothshift=0.)
    
    im.mirror(X=2,Y=2,nbpix=0,color=[255]*3)
    im.sin()
    im.transform()
    return True

if __name__ == "__main__":
    pytest.main()