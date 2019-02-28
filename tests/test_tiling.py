# -*- coding: utf-8 -*-
"""Image transformation test meant to be run with pytest."""
import sys

import pytest

from confmap import HyperbolicTiling

sys.path.append("tests")

def test_tiling():
    HT=HyperbolicTiling('./examples/sample1.png',prefix='./examples/',suffix='1',
                        output_width=900,output_height=900)
    HT.transform(c=0.95,d=0.+0.0j,backcolor=[97/255,102/255,104/255],vanishes=False,
                nbit=25,delta=1e-3,print_and_save=True)
    return True

if __name__ == "__main__":
    pytest.main()