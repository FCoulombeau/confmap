import numpy as np
import matplotlib.pyplot as plt
from .functions import somme,sign
from .mapping import ComplexMapping

eps=1e-7
eps1=1e-5

#def sign(x):
#    """Returns 1 if the argument is positive, -1 else"""
#    return (x>=0)*2-1

def C(n,p=None,memoire=[]):
    if n<len(memoire):
        if p:
            return memoire[n][p]
        return memoire[n]
    for k in range(n-len(memoire)+1):
        if len(memoire)==0:
            memoire.append([1])
        else:
            memoire.append([1]+[memoire[-1][i]+memoire[-1][i+1] for i in range(len(memoire[-1])-1)]+[1])
    if p:
        return memoire[n][p]
    return memoire[n]

#def somme(l):
#    """Takes an array l and returns the array made of the sums of all subarrays
#    over the first dimension along."""
#    res=l[0]
#    for k in l[1:]: res+=k
#    return res

def _droite(a,b,zero):
    assert abs(b-a)>eps
    dt=((b-a)*a.conjugate()).imag
    if abs(dt)>eps:
        x=(-(abs(b)**2-abs(a)**2)*a.imag+(1+abs(a)**2)*(b-a).imag)/2/dt
        y=-(-(abs(b)**2-abs(a)**2)*a.real+(1+abs(a)**2)*(b-a).real)/2/dt
        c=x+1j*y
        r=(abs(c)**2-1)**0.5
        f=lambda z:np.abs(z-c)-r
        if f(zero)<0:
            f=lambda z:-np.abs(z-c)+r
    else:
        c=(b-a)/abs(b-a)
        r=np.Infinity
        f=lambda z:(z*c.conjugate()).imag
        if f(zero)<0:
            f=lambda z:-(z*c.conjugate()).imag
    return c,r,f

def _reflexion(a,b):
    C,R,D=_droite(a,b,0)
    if R!=np.Infinity:
        res=lambda z:R**2*(z-C)/abs(z-C)**2+C
    else:
        res=lambda z:z.conjugate()*C**2
    return res

def _translation(theta,e):
    """Returns the translation of Poincaré disk, in direction exp(1j*theta),
    with magnitude e"""
#    assert e<1
    return lambda z:(e+z*np.exp(-1j*theta))/(1+e*z*np.exp(-1j*theta))*np.exp(1j*theta)

def _rotation(c,theta):
    """Returns the rotation of Poincaré disk, around c,
    with angle theta"""
#    assert abs(c)<1 or (abs(c)==1 and theta%(2*np.pi)!=np.pi)
    if abs(c)==1:
        return lambda z:((1j*(1+np.cos(theta))-np.sin(theta))*(z*(-c.conjugate()))-np.sin(theta))/\
                        (np.sin(theta)*(z*(-c.conjugate()))+np.sin(theta)+1j*(1+np.cos(theta)))\
                        *(-c)
    else:
        if c!=0:
            th=np.arctan2(c.imag,c.real)
            e=abs(c)
            return lambda z:_translation(th,e)(_translation(th+np.pi,e)(z)*np.exp(1j*theta))
        else:
            return lambda z:z*np.exp(1j*theta)

def getVertices(*symetries):
        n=len(symetries)
        angles=[np.pi/a for a in symetries]
        if sum(angles)>=(n-2)*np.pi/2 or sum([k<0 for k in angles])>0:
            return None
        r=1
        
        c=False
        d=False
        while True:
            s=sum([np.arcsin(r*np.cos(a)/np.sqrt(1+r**2)) for a in angles])
            if s>np.pi:
                r-=1e-1
                if d:
                    break
                c=True
            else:
                r+=1e-1
                if c:
                    break
                d=True
        if c:
            rr=r-1e-1
        else:
            rr=r
            r=rr+1e-1
        while abs(s-np.pi)>eps1 and abs(r-rr)>1e-15:
            rrr=(r+rr)/2
            s=sum([np.arcsin(rrr*np.cos(a)/np.sqrt(1+rrr**2)) for a in angles])
            if s>np.pi:
                r=rrr
            else:
                rr=rrr
        angsom=[np.arcsin(r*np.cos(a)/np.sqrt(1+r**2)) for a in angles]
        rho=[np.cos(angles[k]+angsom[k])*r/np.sin(angsom[k]) for k in range(n)]
        Vert=[rho[k]*np.exp(1j*sum([angsom[i]+angsom[(i+1)%n] for i in range(k)])) for k in range(n)]
        return Vert

class HyperbolicTiling(ComplexMapping):
    """Defines usefull methods to make hyperbolic tilings filled with an image.
    """
    def __init__(self,name,suffix=0,prefix='',
                 output_width=900,
                 output_height=900,data=None):
        """Creates an object either from a file if data==None or from a numpy
        array given in data.
        name is the name of the input file and will be suffixed by suffix to
        get the name of the output file.
        data allows to give the input as a numpy array instead of a file.        
        """
#        assert 1/P+1/N<1/2
        
        super().__init__(name,suffix=suffix,prefix=prefix,
                         output_width=output_width,output_height=output_height,
                         data=data)
        
#        self.a1=np.pi*(1/P+1/N)
#        self.a2=np.pi/N
#        self.N=N
#        self.P=P
#        
        self._pre=[]
        self._post=[]
#    
#    def setN(self,N):
#        self.a1=np.pi*(1/self.P+1/N)
#        self.a2=np.pi/N
#        self.N=N
#        
#    def setP(self,P):
#        self.a1=np.pi*(1/P+1/self.N)
#        self.P=P
    
    def pre(self,fonc):
        """Add a transformation to be computed on the tiling only."""
        self._pre.append(fonc)
    
    def post(self,fonc):
        """Add a transformation to be computed on the tiling and filling images."""
        self._post.append(fonc)
    
    def getTilingParam(self,N,P):
        a1=np.pi*(1/P+1/N)
        a2=np.pi/N
        r=1/((np.sin(a1)+np.cos(a1)/np.tan(a2))**2-1)**0.5*np.cos(a1)/np.sin(a2)
        sommets=[r*np.exp(2j*k*a2) for k in range(N)]
        return a1,a2,r,sommets
    
        
    def reflexion(self,a,b,pre=True):
        assert abs(a)<=1 and abs(b)<=1
        if pre:
            self.pre(_reflexion(a,b))
        else:
            self.post(_reflexion(a,b))
    
    def translation(self,c,theta=None,pre=True):
        assert abs(c)<1
        if theta is None:
            theta=np.arctan2(c.imag,c.real)
            c=abs(c)
        if pre:
            self.pre(_translation(theta,c))
        else:
            self.post(_translation(np.pi+theta,c))
    
    def rotation(self,c,theta,pre=True):
        assert abs(c)<=1
        if abs(c)==1 and theta%(2*np.pi)==np.pi:
            theta=0
        if pre:
            self.pre(_rotation(c,theta))
        else:
            self.post(_rotation(c,-theta))

    def _couleur(self,c,lpu,ltr,IM,nbit,cc=1.,dX=0.,dY=0.,delta=0.,pipe=False):
        cX=cc
        cY=cc/IM.shape[1]*IM.shape[0]
        res=np.zeros((c.shape[0],c.shape[1],IM.shape[2]))
#        th,n=trans
#        T=C(n)
#        r=sum([T[2*k+1]*(-dtrans)**(2*k+1) for k in range(len(T)//2)])/sum([T[2*k]*(-dtrans)**(2*k) for k in range((len(T)+1)//2)])
#        c=_translation(th,r)(np.copy(c))
        b=(np.abs(c)<1)
        tY=IM.shape[0]
        tX=IM.shape[1]
        
        i=1
        while i<=nbit and b.any():
            i+=1
            bb=np.copy(b)
            b2=np.zeros(b.shape,dtype=np.bool)
            for j in range(len(lpu)):
                pu=lpu[j]
                tr=ltr[j]
                b2[b]=b[b]&(pu(c[b])<-delta)
                c[b2]=tr(c[b2])

            for pu in lpu:
                bb=bb&(pu(c)>=delta)
            b=b&(~bb)
            res[bb]=IM[np.int16((1-np.imag(c[bb])/cY-dY)*tY/2)%IM.shape[0],np.int16((1+np.real(c[bb])/cX+dX)*tX/2)%IM.shape[1]]
        if pipe:
            return c
        return res
    
    def transform(self,c=1.,d=0.,backcolor=[0,0,0],vanishes=False,print_and_save=True,
              coloris=np.pi/5,nbit=5,delta=0.005,sommets=(5,4)):
        
        if len(sommets)==2:
            N,P=sommets
            assert 1/P+1/N<1/2
            a1=np.pi*(1/P+1/N)
            a2=np.pi/N
            r=1/((np.sin(a1)+np.cos(a1)/np.tan(a2))**2-1)**0.5*np.cos(a1)/np.sin(a2)
            sommets=[r*np.exp(2j*(k/N)*np.pi) for k in range(N)]
            self.vertices=sommets
        else:
            N=len(sommets)
            sommets=getVertices(*sommets)
            self.vertices=sommets
            
        im2=self.data
        
        tY,tX=im2.shape[0:2]
        X=self._width
        Y=self._height
        fond=np.ones((Y,X,im2.shape[2]),dtype=im2.dtype)
        if self._left is None :
            if X>Y:
                ym=-1
                yM=1
                xm=-X/Y
                xM=X/Y
            elif X<Y:
                ym=-Y/X
                yM=Y/X
                xm=-1
                xM=1
            else:
                yM=xM=1
                ym=xm=-1
            grid=np.mgrid[ym:yM:(Y*1j),xm:xM:(X*1j)]
            z=(grid[0]*1j+grid[1]).conjugate()
        else:
            z=self._left.transform()
            
        ind=abs(z)>=1
        if isinstance(backcolor,list):
            for k in range(3):
                fond[:,:,k]=fond[:,:,k]*backcolor[k]
        elif backcolor:
            for k in range(3):
                truc=np.fromfunction(lambda x,y:(k==0)*200*np.cos(x/fond.shape[0]*np.pi/2)*\
                                                  np.cos(y/fond.shape[1]*np.pi/2-coloris)**2+\
                                                  (k==1)*200*np.sin(x/fond.shape[0]*np.pi/2)*\
                                                  np.cos(y/fond.shape[1]*np.pi/2-coloris)**2+\
                                                  (k==2)*(100+100*np.sin(y/fond.shape[1]*np.pi/2-coloris)),
                                                  fond.shape[:2])
                if fond.dtype==np.float32:
                    divisor=255
                else:
                    divisor=1
                fond[:,:,k][ind]=eval('np.'+str(fond.dtype))(truc/divisor)[ind]
        else:
            fond=0*fond
        
        zero=0
        for f in self._pre:
            sommets=[f(k) for k in sommets]
            zero=f(zero)
        LPu=[]
        LTr=[]
        for k in range(N):
            try:
                pu=_droite(sommets[k],sommets[(k+1)%N],zero)[2]
                tr=_reflexion(sommets[k],sommets[(k+1)%N])
                LPu.append(pu)
                LTr.append(tr)
            except Exception:
                pass
        for f in self._post[::-1]:
            z=f(z)
        im=self._couleur(z,LPu,LTr,im2,nbit,cc=c,dX=d.real,dY=d.imag,
                         delta=delta*(1-abs(zero)**2),pipe=(self._right is not None))
        if self._right is not None:
            return z
        if vanishes:
            for k in range(3):
                im[:,:,k]=eval('np.'+str(im2.dtype))(abs(z)**6+im[:,:,k]*(1-abs(z)**6))
        else:
            im=eval('np.'+str(im2.dtype))(im)
        im[ind]=fond[ind]
        if print_and_save:
            plt.imsave(self.prefix+self.name+'-'+str(self.suffix)+'.'+self.format,im)
            plt.imshow(im)
        return im