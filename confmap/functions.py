import numpy as np

__all__ = ["normcar", "arg", "ln", "scale_rotate", "somme", "mirror", "solve",
           "morphing", "halfhalf", "_power", "_second_degree_polynomial",
           "_inverse_polynomial", "_complex_exponential", "_complex_sine",
           "_complex_sine2", "_complex_cosine", "_complex_cosine2",
           "_squareroot_sine", "_complex_tangent", "_symmetry4", "_symmetry4v2",
           "_symmetry3", "_symmetryn", "_complex_arcsine", "_complex_arccos",
           "_complex_arctan", "_complex_logarithm", "_oval", "_semi_scaling",
           "_fisheye", "_invers_fisheye", "_equirectangular", "_spherize",
           "_flatten", "_expression", "_compose"]

normcar=lambda u:u.real**2+u.imag**2
arg=lambda u:np.arctan2(u.imag,u.real)
ln=lambda u:np.log(normcar(u))/2.+1j*arg(u)
scale_rotate=lambda r,t:r*(np.cos(t*2*np.pi)+1j*np.sin(t*2*np.pi))

def sign(x):
    """Returns 1 if the argument is positive, -1 else"""
    return (x>=0)*2-1

def mirror(d,X=2,Y=2,nbpix=0,color=[255,255,255]):
    """Returns the numpy array representing the output image got by mirroring
    the input d along its right edge if X=2 and along its bottom edge if Y=2.
    Puts stripes with the number "nbpix" of pixels around the output and
    between the mirrored inputs with color "color".
    """
    li,co=d.shape[0:2]
    r=np.ndarray((li*Y+(Y+1)*nbpix,X*co+(X+1)*nbpix,d.shape[2]),dtype=d.dtype)
    r[:nbpix,:]=color
    r[:,:nbpix]=color
    r[li+nbpix:li+2*nbpix,:]=color
    r[:,co+nbpix:co+2*nbpix]=color
    r[nbpix:li+nbpix,nbpix:co+nbpix]=d[:,:]
    if Y==2:
        if nbpix>0:
            r[li+2*nbpix:-nbpix,:]=r[li+nbpix-1:(2-Y)*(li+nbpix)+nbpix-1:-1,:]
            r[-nbpix:,:]=color
        else:
            r[li:,:]=r[li-1::-1,:]
    if X==2:
        if nbpix>0:
            r[:,co+2*nbpix:-nbpix]=r[:,co+nbpix-1:(2-X)*(co+nbpix)+nbpix-1:-1]
            r[:,-nbpix:]=color
        else:
            r[:,co:]=r[:,co-1::-1]
    return r
    
def solve(obs,f,ff,u,mx=50):
    d=normcar(u)**0.5
    x=d
    xx=x-(f(x)+obs/d*x-obs)/(ff(x)+obs/d)
    i=0
    while (np.max(np.abs(x-xx))>1e-4)and(i<mx):
        i+=1
        x=xx
        xx=x-(f(x)+obs/d*x-obs)/(ff(x)+obs/d)
    return x
    
def morphing(im1,wt1,im2,wt2):
    """Returns a mix of the images im1 with weight wt1 and im2 with weight wt2,
    each image given as a numpy array.
    """
    return np.uint8((np.uint16(im1[:,:,:])*wt1+np.uint16(im2[:,:,:])*wt2)/(wt1+wt2))

def halfhalf(im1,im2):
    """Returns the image made of half im1 and half im2, horizontally.
    im1 and im2 must have the same height, and im2 must be wider than im1.
    """
    im=np.ndarray(im1.shape,np.uint8)
    im[:,:im1.shape[1]//2,:]=im1[:,:im1.shape[1]//2,:]
    im[:,im1.shape[1]//2:,:]=im2[:,im1.shape[1]//2:,:]
    return im

def _change_origin(foncInv,c,d):
    return lambda u:foncInv(u)/c+d

def somme(l):
    """Takes an array l and returns the array made of the sums of all subarrays
    over the first dimension along."""
    res=l[0]
    for k in l[1:]: res+=k
    return res

def _power(Exposant,c=1.,d=0.,rel=1.):
    return (lambda u:np.power(normcar(u),1./2/Exposant)*
            (np.cos(arg(u)/Exposant)+1j*np.sin(arg(u)/Exposant)),c,d)

def _second_degree_polynomial(a,b,c=1.,d=0.):
    return (lambda u:normcar(a**2+4*(b-u))**(1./4)*(np.cos(arg(a**2+4*(b-u))/2)+1j*np.sin(arg(a**2+4*(b-u))/2)),c,d)

def _inverse_polynomial(l,c=1.,d=0.):
    return (lambda u:somme([l[k]*np.power(u,k) for k in range(len(l))]),c,d)

def _complex_exponential(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):    
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)    
        c2=(-1)**(Q//2)*np.pi/P*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (ln,c2,d)

def _complex_sine(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:ln(u+_power(2,c2,d)[0](1+u**2)),c2,d)
    
def _complex_sine2(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:ln(u-_power(2,c2,d)[0](1+u**2)),c2,d)

def _complex_cosine(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:ln(u+_power(2,c2,d)[0](u**2-1)),c2,d)

def _complex_cosine2(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:ln(u-_power(2,c2,d)[0](u**2-1)),c2,d)

def _squareroot_sine(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:ln(u**2+_power(2,c2,d)[0](1+u**4)),c2,d)

def _complex_tangent(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)    
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:(-ln(1-u)+ln(1+u))/2,c2,d)

def _symmetry4(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:(-ln(1-u)+ln(1+u)-ln(1j-u)+ln(1j+u))/2,c2,d)
    
def _symmetry4v2(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:(ln(-1+u)+ln(1+u)+ln(1j+u)+ln(-1j+u))/2,c2,d)
    
def _symmetry3(forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:(ln(1+u)+ln(-1/2+np.sqrt(3)/2*1.j+u)+ln(-1/2-np.sqrt(3)/2*1.j+u))/2,c2,d)

def _symmetryn(n,forme,N=1,P=1,Q=0,Angle=True,c=1.,d=0.):
    if Angle:
        alpha=np.arctan2((-1)**Q*P,N*forme)
        c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    else:
        c2=c
    return (lambda u:(sum([ln(np.cos(k*2*np.pi/n)+1.j*np.sin(k*2*np.pi/n)+u) for k in range(n)]))/2,c2,d)
    
def _complex_arcsine(c=1.,d=0.):
    return (lambda u:np.sinh(u.real)*np.cos(u.imag)+1j*np.sin(u.imag)*np.cosh(u.real),c,d)

def _complex_arccos(c=1.,d=0.,mod=0):
    return (lambda u:np.cosh(u.real)*np.cos(u.imag)+1j*np.sin(u.imag)*np.sinh(u.real),c,d)

def _complex_arctan(c=1.,d=0.,hyperb=False,dec=0):
    if hyperb:
        return (lambda u:(np.tanh(u.real+((u.imag+np.pi/2)//np.pi-1/2)*dec)+1j*np.tan(u.imag))/(1+np.tanh(u.real+((u.imag+np.pi/2)//np.pi-1/2)*dec)*1j*np.tan(u.imag)),c,d)
    else:
        return (lambda u:(np.tanh(u.real)+1j*np.tan(u.imag))/(1+np.tanh(u.real)*1j*np.tan(u.imag)),c,d)
    
def _complex_logarithm(c=1.,d=0.):
    return (lambda u:np.exp(u.real)*(np.cos(u.imag)+1j*np.sin(u.imag)),c,d)
    
def _oval(forme,N=1,P=1,Q=0,d=0.):
    alpha=np.arctan2((-1)**Q*P,N*forme)    
    c2=(-1)**(Q//2)*np.pi/P/2*np.sin(alpha)*(1.j*np.cos(alpha)+np.sin(alpha))
    cc2=np.sinh(np.pi*np.cos(np.pi/2-2*alpha)/2)*np.cos(np.pi*np.sin(np.pi/2-2*alpha)/2)+1j*np.cosh(np.pi*np.cos(np.pi/2-2*alpha)/2)*np.sin(np.pi*np.sin(np.pi/2-2*alpha)/2)
    g2=lambda u:ln(u+_power(2,c2,d)[0](1+u**2))
    return (lambda u:g2(g2(u)*cc2/1j*2/np.pi),c2,d)

def _semi_scaling(r):
    return lambda u:u.real/r+1j*u.imag

def _fisheye(aff=1.,c=1.,d=0.):
    return (lambda u:_semi_scaling(1/aff)(_power(2)[0](_complex_sine(1.,Angle=False)[0](_power(1/2)[0](_semi_scaling(aff)(u))*np.pi/4)/np.pi*4)),c,d)

def _invers_fisheye(aff=1.,c=1.,d=0.):
    return (lambda u:_semi_scaling(1/aff)(_power(2)[0](_complex_arcsine()[0](_power(1/2)[0](_semi_scaling(aff)(u))*np.pi/4)/np.pi*4)),c,d)

def _equirectangular(c=1.,d=0.):
    return (lambda u:np.tan(np.pi*u.imag/4+np.pi/4)*(1j*np.cos(np.pi*u.real/2)+np.sin(np.pi*u.real/2)),c,d)

def _spherize(c=1.,d=0.):
    return (lambda u:u/((1-normcar(u)))**(1/2),c,d)

def _flatten(c=1.,d=0.):
    return (lambda u:u/np.sqrt(normcar(u)+1),c,d)

def _expression(c=1.,d=0.,express="z"):
    return (lambda z:eval(express),c,d)

def _compose(function, *funcs):
    if funcs:
        return lambda u:_change_origin(*function)(_compose(*funcs)(u))
    else:
        return lambda u:_change_origin(*function)(u)