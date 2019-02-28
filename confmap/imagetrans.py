"""Conformal mappings of the complex plane used to manipulate and transform
images and videos.
@author: François Coulombeau
@date : 2019-02
@version : 1.0
@changes : Structured as a module.   
"""

from .functions import *

import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import time
from .mapping import ComplexMapping

__all__ = ["normcar", "arg", "ln", "scale_rotate", "Mask", "ImageTransform", "VideoTransform"]

class ImageTransform(ComplexMapping):
    """Defines usefull methods to make conformal mappings over an image.
    The input image is put between -1j and 1j with origin in the middle of the
    image (real part depends on the shape of the input).
    The output image is also put between -1j and 1j with origin in the middle 
    of the image (real part depends on the shape of the output).
    """
    def __init__(self,name,suffix=0,prefix='',output_width=1024,output_height=704,
                 data=None,r=1.,c=1.,d=0.,blur=False,shift=0,smoothshift=0.):
        """Creates an object either from a file if data==None or from a numpy
        array given in data.
        name is the name of the input file and will be suffixed by suffix to
        get the name of the output file.
        d allows to translate the output.
        c scales and rotate the output before translation.
        r scales and rotate the output after translation.
        blur allows to blur the output.
        data allows to give the input as a numpy array instead of a file.        
        """
        
        super().__init__(name,suffix=suffix,prefix=prefix,
                         output_width=output_width,output_height=output_height,
                         data=data)
        
        self._input_scaling = self.data.shape[1]/self.data.shape[0]
        self._output_scaling = self._width/self._height
        self._output_rect = [-self._output_scaling-1.j,self._output_scaling+1.j]
        self.c = r*c
        self.d = r*(d.real*self._output_scaling+1j*d.imag)
        self.blur = blur
        self._transformations=[]
        self._shift=shift
        self._sshift=smoothshift
        
    def mirror(self,X=2,Y=2,nbpix=0,color=[255,255,255]):
        """Replaces the input image by a mirrored version got by mirroring
        the input d along its right edge if X=2 and along its bottom edge if Y=2.
        Puts stripes with the number "nbpix" of pixels around the output and
        between the mirrored inputs with color "color".
        """
        if self.data.dtype==type(np.float32(1.)):
             color=[color[k]/255 for k in range(3)]
             if self.data.shape[2]==4:
                 color = color+[1.]
        self.data = mirror(self.data,X=X,Y=Y,nbpix=nbpix,color=color)
        self._input_scaling = self.data.shape[1]/self.data.shape[0]
    def _f(self,z):
        u=(z+self.d)/self.c       
        if self._transformations:
            return _compose(*self._transformations)(u)
        else:
            return u
    def nouvtrans(self,fonc,c,d):
        """Add a transformation to be computed."""
        self._transformations+=[(fonc,c,d)]
        
    def similitude(self,c=1.,d=0.,auto=True):
        """Mapping z->c*(z-d)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(lambda u:u,c,d)
        
    def cut(self,rect=[-1.-1.j,1.+1.j],c=1.,d=0.,auto=True):
        """Fill the plane with copies of the given rectangle rect.
        If auto==True, the rectangle is the output image seen before the cut
        and complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            rect=[-self._output_scaling-1.j,self._output_scaling+1.j]
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(lambda u:rect[0]+(u.real-rect[0].real)%(rect[1].real-rect[0].real)+1.j*((u.imag-rect[0].imag)%(rect[1].imag-rect[0].imag)),c,d)
    
    def power(self,exponent,c=1.,d=0.,auto=True):
        """Mapping z->(c*(z-d))**exponent
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_power(exponent,c,d))
        
    def polsec(self,a,b,c=1.,d=0.,auto=True):
        """Mapping z->(c*(z-d))**2+a*c*(z-d)+b
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_second_degree_polynomial(a,b,c,d))
    
    def invpol(self,l,c=1.,d=0.,auto=True):
        """Reverse mapping of z->sum(l[i]*(c*(z-d))**i)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_inverse_polynomial(l,c,d))
    
    def exp(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->exp(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_exponential(forma,N,P,Q,angle,c,d))
    def sin(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->sinh(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_sine(forma,N,P,Q,angle,c,d))
    
    def sin2(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->sinh(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_sine2(forma,N,P,Q,angle,c,d))
    
    def cos(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->sinh(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_cosine(forma,N,P,Q,angle,c,d))
    
    def cos2(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->sinh(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_cosine2(forma,N,P,Q,angle,c,d))
            
    def tan(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """Mapping z->tanh(c*(z-d))
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_complex_tangent(forma,N,P,Q,angle,c,d))
    
    def symmetry4(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """A mapping built on tangent function with symmetry 4.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_symmetry4(forma,N,P,Q,angle,c,d))
    
    def symmetry4_v2(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """A mapping built on tangent function with symmetry 4.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_symmetry4v2(forma,N,P,Q,angle,c,d))
    
    def symmetry3(self,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """A mapping built on tangent function with symmetry 3.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_symmetry3(forma,N,P,Q,angle,c,d))
    
    def symmetryn(self,n,form=None,N=1,P=1,Q=0,angle=True,auto=True,c=1.,d=0.):
        """A mapping built on tangent function with symmetry 3.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        The value of c isn't used in this case but is computed using the values
        of form, N, P, Q and angle.
            * form is the scaling (width/height) of the transformation input;
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input;
            * angle determines whether the transformation will result in 
            spirals (angle=True) or circles (angle=False)
        """
        if form:
            forma=form
        else:
            if self._transformations==[]:
                forma=self._input_scaling
            else:
                forma=self._output_scaling
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
            c=np.pi/2/N*(1.j)**Q*(Q%2/forma+(Q+1)%2)
        self.nouvtrans(*_symmetryn(n,forma,N,P,Q,angle,c,d))
    
    def arcsin(self,auto=True,c=1.,d=0.):
        """Mapping z->asin(z)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        Furthermore, the output is then reduced by a factor pi/2.
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_complex_arcsine(c,d))
        if auto:
            self.similitude(c=2/np.pi)
    
    def arccos(self,auto=True,c=1.,d=0.):
        """Mapping z->asin(z)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        Furthermore, the output is then reduced by a factor pi/2.
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_complex_arccos(c,d))
        if auto:
            self.similitude(c=2/np.pi)
        
    def arctan(self,auto=True,c=1.,d=0.):
        """Mapping z->atan(z)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        Furthermore, the output is then reduced by a factor pi/2.
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_complex_arctan(c,d))
        if auto:
            self.similitude(c=2/np.pi)
        
    def ln(self,auto=True,c=1.,d=0.):
        """Mapping z->ln(z)
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        Furthermore, the output is then reduced by a factor pi.
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_complex_logarithm(c,d))
        if auto:
            self.similitude(c=1/np.pi)
        
    def oval(self,N=1,P=1,Q=0,auto=True,d=0.):  
        """A mapping built on sine function removing the corners of the input.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
            * N is the number of copies of the input width-wise;
            * P is the number of copies of the input height-wise;
            * Q is the number of quarter of turn applied to the input.
        """
        if self._transformations==[]:
            forma=self._input_scaling
        else:
            forma=self._output_scaling
        if auto:
            d=d.real*forma+1j*d.imag
        self.nouvtrans(*_oval(forma,N,P,Q,d))
        if auto:
            self.similitude(c=(-1-1.j)/2**0.5)
    def fisheye(self,form=1.,c=1.,d=0.,auto=True):
        """A mapping built on sine function simulating fisheye.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
            * form is the scaling (width/height) of the transformation input.
        """
        if auto:
            if self._transformations==[]:
                form=self._input_scaling
            else:
                form=self._output_scaling
            d=d.real*form+1j*d.imag
        self.nouvtrans(*_fisheye(form,c,d))
    
    def invers_fisheye(self,form=1.,c=1.,d=0.,auto=True):
        """A mapping built on sine function simulating invers fisheye.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
            * form is the scaling (width/height) of the transformation input.
        """
        if auto:
            if self._transformations==[]:
                form=self._input_scaling
            else:
                form=self._output_scaling
            d=d.real*form+1j*d.imag
        self.nouvtrans(*_invers_fisheye(form,c,d))
        
    def equirectangular(self,c=1.,d=0.,auto=True):
        """Equirectangular projection allowing 3D panoramas of the whole
        complex plane.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:        
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag-1j/c
            else:
                d=d.real*self._output_scaling+1j*d.imag-1j/c
        self.nouvtrans(*_equirectangular(c,d))
    
    def tchebychev(self,c=1.,d=0.,N=5,auto=True,d2=0):
        """Tchebychev polynomial of degree N.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.arccos(auto=False,d=d,c=c)
        self.sin(auto=False,angle=False,c=N,d=1.j*np.pi/2+d2)
        self.similitude(c=-1j)
        if auto:        
            self.similitude(c=1/2**(N-1))
    
    def tchebychevb(self,c=1.,d=0.,N=5,auto=True):
        """Tchebychev polynomial of degree N.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.arccos(auto=False,d=d,c=-c)
        self.sin(auto=False,angle=False,c=N,d=-1.j*np.pi/2)
        self.similitude(c=1j)
        if auto:        
            self.similitude(c=1/2**(N-1))
        
    def tchebychev2(self,c=1.,d=0.,N=5, auto=True):
        """Tchebychev polynomial of the second kind of degree N.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.arcsin(auto=False,d=d,c=c)
        self.sin(auto=False,angle=False,c=N,d=0.)
        if auto:        
            self.similitude(c=1/2**(N-1))
            
    def spherize(self,c=1.,d=0.,auto=True):
        """Maps the whole complex plane onto a half-sphere and projects it back 
        to unit disk.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_spherize(c,d))

    def flatten(self,c=1.,d=0.,auto=True):
        """Maps the unit disk onto a half-sphere and projects it back 
        to the whole complex plane.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_flatten(c,d))
        
    def expression(self,c=1.,d=0.,auto=True,express="z"):
        """Maps the complex plane using z->express^{-1} where express must be 
        a string representing a valid expression of the variable z.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(*_expression(c,d,express))
    
    def maps(self,f=lambda z:z,ff=lambda z:1,obs=1.,c=1.,d=0.,auto=True):
        """Maps the complex plane onto a symmetrical surface given by its 
        equation height=f(rho) and projects it back to the complex plane using 
        the observer position (0,0,obs). f and its derivative ff must be given.
        If auto==True, complex number d is expressed in percent of the 
        input shape before transformation (which is the output shape
        if the transformation isn't the first).
        """
        if auto:
            if self._transformations==[]:
                d=d.real*self._input_scaling+1j*d.imag
            else:
                d=d.real*self._output_scaling+1j*d.imag
        self.nouvtrans(lambda u:u*solve(obs,f,ff,u)/normcar(u)**0.5,c,d)
        
    def _barycentre(self,indices):
            iu,iv=indices.real,indices.imag
            if self.blur:
                u,v=iu%self.data.shape[1],iv%self.data.shape[0]
                uu,vv=np.int32(np.floor(u)),np.int32(np.floor(v))
                du,dv=u-uu,v-vv
                coef=np.array([(du if i else 1-du)*(dv if j else 1-dv) for i in range(2) for j in range(2)])
                typ=type(self.data[0,0,0])

                coef=np.array(list(zip(*([coef]*self.data.shape[2])))).reshape(4,self._height,self._width,self.data.shape[2])
                try:
                    cl=typ(coef[0]*self.data[[vv],[uu]][0]+coef[1]*self.data[[(vv+1)%self.data.shape[0]],[uu]][0]+coef[2]*self.data[[vv],[(uu+1)%self.data.shape[1]]][0]+coef[3]*self.data[[(vv+1)%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0])
                except Exception:
                    print("Strange error :",str(self._transformations))
                    cl=typ(coef[0]*self.data[[vv%self.data.shape[0]],[uu%self.data.shape[1]]][0]+coef[1]*self.data[[(vv+1)%self.data.shape[0]],[uu%self.data.shape[1]]][0]+coef[2]*self.data[[vv%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0]+coef[3]*self.data[[(vv+1)%self.data.shape[0]],[(uu+1)%self.data.shape[1]]][0])

                return cl
            else:
                xx=np.array(np.floor(iu%self.data.shape[1]),dtype=int)
                x=np.array(np.floor(iu//self.data.shape[1]),dtype=int)
                try:
                    cl2=np.array(np.floor((iv+x*self._shift+xx*self._sshift)%self.data.shape[0]),dtype=int)
                    cl=self.data[[cl2],[xx],:]
                except Exception:
                    print("Strange error :",str(self._transformations))
                    cl2=np.array(np.floor(iv+x*self._shift+xx*self._sshift)%self.data.shape[0],dtype=int)
                    cl=self.data[[cl2],[xx],:]
                return cl[0]    
    
    def _trouveCouleurs(self,infini=True,liste=[[0,0]],MaxX=np.NaN,
                        color=[255]*3, fade=False, fade_factor=1,
                        fade_color=[255]*3,fade_pos=0.,MinY=np.nan):
        bx,by,ex,ey=self._output_rect[0].real,self._output_rect[0].imag,self._output_rect[1].real,self._output_rect[1].imag
        if self._left is None:
            grid=np.mgrid[by:ey:(self._height*1j),bx:ex:(self._width*1j)]
            grid=(grid[0]*1j+grid[1]).conjugate()
        else:
            grid=self._left.transform()
        FPt=self._f(grid)
        if self._right is not None:
            return np.nan_to_num(FPt)
        Pt=np.nan_to_num((FPt+(self._input_scaling+1j))*self.data.shape[0]/2)
        res=self._barycentre(Pt)
        if fade:
            a=np.exp(-fade_factor*(normcar(FPt-fade_pos.conjugate())))        
            b=np.array([[[k]*3 for k in a[i,:]] for i in range(a.shape[0])])
            if (np.array(fade_color).shape[0]<=4):
                col=np.array(fade_color*a.shape[0]*a.shape[1]).reshape(a.shape[0],a.shape[1],3)
                truc=col+(np.int32(res[:,:,:3])-np.int32(col))*b
                res[:,:,:3]=np.uint8(truc)
            else:
                res=np.uint8(fade_color+(res-fade_color)*b)
        if not(np.isnan(MaxX)):
            indices=np.array([[np.floor(j.real/self.data.shape[1])>MaxX for j in i] for i in Pt])
            res[indices]=color
        if not(np.isnan(MinY)):
            indices=np.array([[np.floor(j.imag/self.data.shape[0]/2)>MinY for j in i] for i in Pt])
            res[indices]=color
        if infini:
            return res
        if res.shape[2]==4:
            color = color+[1.]
        if res.dtype==type(np.float32(1.)):
             color=[color[k]/255 for k in range(3)]
        else:
            color=np.uint8(color)
        indices = np.array([[[np.floor(j.real/self.data.shape[1]),
                              np.floor((j.imag+self._shift*np.floor(j.real/self.data.shape[1])+self._sshift*np.floor(j.real%self.data.shape[1]))/self.data.shape[0])]
                              not in liste for j in i] for i in Pt])
        res[indices] = color
        return res

    def transform(self, infinite=True, lst=[[0,0]], MaxX=np.NaN, color=[255]*3,
                  print_and_save=True,fade=False,fade_factor=1,
                  fade_color=[255]*3,fade_pos=0.,verbose=False,MinY=np.nan):
        """Perform the transformation(s).
        If infinite=True, the whole plane is covered by the input image or half
        the plane is MaxX is given.
        Else, the input is placed on places given in lst.
        If print_and_save=True, the ouput is automatically printed and saved.
        In any case, returns the array containing the output.
        """
        start = time.time()
        Couleurs=self._trouveCouleurs(infini=infinite,liste=lst,color=color,
                                      MaxX=MaxX, fade=fade, 
                                      fade_factor=fade_factor,
                                      fade_color=fade_color, fade_pos=fade_pos,
                                      MinY=MinY)
        if verbose:print("Calcul :",time.time()-start)
        if print_and_save and (self._right is None):
            plt.imshow(list(Couleurs), cmap=plt.cm.gray)
            plt.imsave(self.prefix+self.name+"-"+str(self.suffix)+'.'+self.format,Couleurs)
        return Couleurs

    def video(self,trans,nbim,filename,gif=True,infinite=True,lst=[[0,0]],
              MaxX=np.NaN,color=[255]*3,pause=0,fps=10):
        """Makes a gif or a video from an input image applying transformations
        given in trans parameter which evolve according to the integer value of
        a variable called i.
        trans must be a string containing the transformations separated by ;
        nbim is the number of images of the output.
        filename is the name of the file saved.
        If gif is True, the output is a GIF.
        """
        imgs=[]
        s=trans.split(";")
        for i in range(nbim):
            for t in s:
                exec("self."+t)
            start = time.time()
            Couleurs=self._trouveCouleurs(infini=infinite,liste=lst,color=color,MaxX=MaxX)
            print("Image "+str(i)+" - Calcul :",time.time()-start)
            self._transformations=[]
            imgs.append(Couleurs)
        for i in range(fps*pause):
            imgs.append(Couleurs)
        v=mpy.ImageSequenceClip(imgs,10,with_mask=False,ismask=False)
        if gif:
            v.to_gif(filename,fps=fps,loop=0)#,program='ImageMagick')
        else:
            v.to_videofile(filename,fps=fps,audio=False)

    def sample(self,rep='./'):
        """Saves transformations sample into the directory rep (current 
        directory by default).
        """
        trans={".invers_fisheye()",".fisheye()",".oval()",".ln()",".arctan()"
               ,".arcsin()",".symmetry4()",".symmetry4_v2()",".symmetry3()"
               ,".tan()",".sin()",".exp()",".tan(angle=False)",".sin(angle=False)"
               ,".exp(angle=False)",".symmetry4(angle=False)",".symmetry4_v2(angle=False)"
               ,".symmetry3(angle=False)",".invpol([0.,-1,1.5,-0.5])",".power(2,c=0.3,d=-1-1j)"
               ,".power(1.33,c=0.3-0.3j,d=-1-1j)",".power(0.5,d=-1-1j)"}
        for k in trans:
            exec("self"+k)
            start = time.time()
            Couleurs=self._trouveCouleurs()
            print(k+" - Calcul :",time.time()-start)
            plt.imsave(rep+self.name+k+'.'+self.format,Couleurs)
            self._transformations=[]