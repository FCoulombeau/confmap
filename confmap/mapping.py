import matplotlib.pyplot as plt
import numpy as np
from .functions import somme

class ComplexMapping:
    """Abstract class meant to be used with HyperbolicTiling and 
    ImageTransform"""
    def __init__(self,name,suffix=0,prefix='',output_width=900,
                 output_height=900,data=None):
        """Takes care of name, data and size initializations"""
        if isinstance(name,ComplexMapping):
            self.data = np.zeros((name._height,name._width,3),dtype=np.uint8)
            self.format = name.format
            self.suffix = name.suffix
            self.prefix = name.prefix
            self.name = name.name
            self._width = name._width
            self._height = name._height
            self._right = name
            name._left = self
            self._left = None
        else:
            if data is not None:
                self.data = data[:,:,:3]
            else:
                self.data = plt.imread(name)[:,:,:3]
            self.suffix = suffix
            self.prefix = prefix
            
            name=name.split('/')[-1]
            self.name = somme(name.split('.')[0:-1])
            self.format = name.split('.')[-1]
            
            self._width = output_width
            self._height = output_height
            self._right = None
            self._left = None

        
        