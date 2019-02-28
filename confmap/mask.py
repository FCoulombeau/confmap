import numpy as np

class Mask:
    """Defines method to create and draw masks."""
    def __init__(self,size,value=255):
        self._image=np.full((size[1],size[0],3),value,dtype=np.uint8)
        self._sx,self._sy=size
    def asimage(self):
        return self._image
    def ellipse(self,center,axes,border,color=False):
        A=(axes[0][0]**2+axes[0][1]**2)**0.5
        assert (A>0)and(axes[1]>0)
        vn=[axes[0][0]/A,axes[0][1]/A]
        a=255/(2*A+max(border,1e-3))/max(border,1e-3)
        c=a*A**2
        b=c/axes[1]**2
        def f(j,i,k):
            x=i-center[0]
            y=j-center[1]
            return np.minimum(255,np.maximum(0,a*(x*vn[0]+y*vn[1])**2+b*(x*vn[1]-y*vn[0])**2-c))
        if color:
            el=255-np.fromfunction(f,(self._sy,self._sx,3))
            indices=el>0
            self._image[indices]=np.array(el,dtype=np.uint8)[indices]
        else:
            
            el=np.fromfunction(f,(self._sy,self._sx,3))
            indices=el<255
            self._image[indices]=np.array(el,dtype=np.uint8)[indices]
    def parallelogram(self,center,axes,border,color=False):
        n1=(axes[0][0]**2+axes[0][1]**2)**0.5
        n2=(axes[1][0]**2+axes[1][1]**2)**0.5
#        f1=(n1+border)/n1
#        f2=(n2+border)/n2
        assert (n1>0)and(n2>0)
        vn1=[axes[0][1]/n1,-axes[0][0]/n1]
        vn2=[axes[1][1]/n2,-axes[1][0]/n2]
        ff1=16/(max(1e-3,border)*(max(1e-3,border)+2*abs(vn1[0]*axes[1][0]+vn1[1]*axes[1][1])))**0.5
        ff2=16/(max(1e-3,border)*(max(1e-3,border)+2*abs(vn2[0]*axes[0][0]+vn2[1]*axes[0][1])))**0.5
        vn1=[axes[0][1]/n1*ff1,-axes[0][0]/n1*ff1]
        vn2=[axes[1][1]/n2*ff2,-axes[1][0]/n2*ff2]
        p11=[center[0]+axes[1][0],center[1]+axes[1][1]]
        p12=[center[0]-axes[1][0],center[1]-axes[1][1]]
        p21=[center[0]+axes[0][0],center[1]+axes[0][1]]
        p22=[center[0]-axes[0][0],center[1]-axes[0][1]]
        def f(j,i,k):
            a=np.minimum(255,np.maximum(0,((vn1[0])*(i-p11[0])+(vn1[1])*(j-p11[1]))*((vn1[0])*(i-p12[0])+(vn1[1])*(j-p12[1]))))
            b=np.minimum(255,np.maximum(0,((vn2[0])*(i-p21[0])+(vn2[1])*(j-p21[1]))*((vn2[0])*(i-p22[0])+(vn2[1])*(j-p22[1]))))
            return np.sqrt((255-a)*(255-b))
        if color:
            el=np.fromfunction(f,(self._sy,self._sx,3))
            indices=el>0
            self._image[indices]=np.array(el,dtype=np.uint8)[indices]
        else:
            el=255-np.fromfunction(f,(self._sy,self._sx,3))
            indices=el<255
            self._image[indices]=np.array(el,dtype=np.uint8)[indices]