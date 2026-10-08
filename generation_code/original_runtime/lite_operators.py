"""Deterministic appearance operators. RGB uint8 in and out; labels unchanged."""
import hashlib
import cv2
import numpy as np

VERSION = 'bdd-lite-v1.0'
FAMILIES = ('low_exposure_noise', 'motion_blur', 'resolution_loss', 'rain_streak')
LEVELS = ('L', 'M', 'H')

def seed_for(*parts):
    return int(hashlib.sha256('|'.join(map(str,(VERSION,2131015654)+parts)).encode()).hexdigest()[:16],16) % (2**32)

def apply_rgb(image, family, level, seed):
    k=LEVELS.index(level); rng=np.random.RandomState(seed); h,w=image.shape[:2]
    if family=='low_exposure_noise':
        x=image.astype(np.float32)/255
        x=np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
        ev=(0.5,1.,1.5)[k]; a=(.0001,.0003,.0006)[k]; b=(1e-6,4e-6,1e-5)[k]
        x=x*(2**(-ev)); x=np.clip(x+rng.standard_normal(x.shape).astype(np.float32)*np.sqrt(a*x+b),0,1)
        x=np.where(x<=.0031308,x*12.92,1.055*x**(1/2.4)-.055)
        return np.rint(np.clip(x,0,1)*255).astype(np.uint8),dict(ev=ev,a=a,b=b,space='linear_rgb')
    if family=='motion_blur':
        length=max(3,int(round((5,9,15)[k]*w/1280))|1); angle=float(rng.uniform(-30,30))
        kernel=np.zeros((length,length),np.float32); c=(length-1)/2; r=c
        dx=r*np.cos(np.deg2rad(angle));dy=r*np.sin(np.deg2rad(angle))
        cv2.line(kernel,(round(c-dx),round(c-dy)),(round(c+dx),round(c+dy)),1.,1,cv2.LINE_8)
        kernel/=kernel.sum()
        return cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_REFLECT_101),dict(length=length,angle=angle)
    if family=='resolution_loss':
        scale=(.75,.5,.35)[k]
        small=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
        return cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR),dict(scale=scale)
    if family=='rain_streak':
        nmax=max(1,round(1800*w*h/1e6)); xy=rng.uniform(0,1,(nmax,2));angle=float(rng.uniform(-15,15))
        length=max(2,round(12*w/1280)); width=max(1,round(w/1280)); n=round((600,1200,1800)[k]*w*h/1e6)
        layer=np.zeros((h,w),np.uint8);dx=round(length*np.sin(np.deg2rad(angle)));dy=round(length*np.cos(np.deg2rad(angle)))
        for x,y in xy[:n]:
            xx=int(x*w);yy=int(y*h);cv2.line(layer,(xx,yy),(xx+dx,yy+dy),255,width,cv2.LINE_AA)
        alpha=(.15,.25,.35)[k]*layer.astype(np.float32)[...,None]/255
        out=image.astype(np.float32)*(1-alpha)+220*alpha
        return np.rint(np.clip(out,0,255)).astype(np.uint8),dict(density=(600,1200,1800)[k],alpha=(.15,.25,.35)[k],angle=angle,length=length,width=width)
    raise ValueError(family)
