from .imagetrans import ImageTransform
import moviepy.editor as mpy
import time as time
import numpy as np

class VideoTransform(ImageTransform):
    """Defines usefull methods to make conformal mappings over a video.
    The input frames are put between -1j and 1j with origin in the middle of the
    frames (real part depends on the shape of the input).
    The output frames are also put between -1j and 1j with origin in the middle 
    of the frames (real part depends on the shape of the output).
    """
    def __init__(self,name,suffix=0,prefix='',output_width=1024,output_height=704,r=1.,c=1.,d=0.,blur=False):
        self.video=mpy.VideoFileClip(name,verbose=True)
        self.duration=self.video.duration
        self.numpict=np.ceil(self.video.duration*self.video.fps)-10
        self.mirrored=False
        ImageTransform.__init__(self,name,suffix,prefix,output_width,output_height,r,c,d,blur,data=self.video.get_frame(0))

    def mirror(self,X=2,Y=2,nbpix=0,color=[255,255,255]):
        self.mirrored=True
        self.X=X
        self.Y=Y
        self.nbpix=nbpix
        self.color=color

    def transform(self,infinite=True,lst=[[0,0]],MaxX=np.NaN,color=[255]*3,
                  verbose=True,test=False): 
        imgs=[]
        N=0
        start = time.time()
        if test:
            nbframes=min(100,self.numpict)
        else:
            nbframes=self.numpict
        for i in range(nbframes):
            if self.mirrored:
                ImageTransform.mirror(self,self.X,self.Y,self.nbpix,self.color)
            imgs.append(ImageTransform.transform(self,print_and_save=False,infinite=infinite,lst=lst,MaxX=MaxX,color=color))
            if (verbose)and(i/nbframes*20>N+1):
                N+=1
                print("Image "+str(i+1)+"/"+str(nbframes)+" - Calcul :",time.time()-start)
                start = time.time()
            self.data=self.video.get_frame(i*self.duration/self.numpict)
        v=mpy.ImageSequenceClip(imgs,self.video.fps,with_mask=False)
        v=v.set_audio(self.video.audio)
        v.to_videofile(self.prefix+self.name+'-'+str(self.suffix)+'.avi',fps=self.video.fps)
