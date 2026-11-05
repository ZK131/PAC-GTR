"""Visible-view aggregation and the frozen native weak-seed update."""
import numpy as np
from scipy.spatial import cKDTree

def visible_view_mean(probabilities, valid):
    p=np.asarray(probabilities,dtype=np.float32)
    ok=np.asarray(valid,dtype=bool)&np.isfinite(p)
    if p.ndim!=2 or ok.shape!=p.shape:raise ValueError('Expected aligned point-by-view arrays')
    count=ok.sum(axis=1)
    return np.divide(np.where(ok,p,0).sum(axis=1),count,out=np.zeros(len(p),dtype=np.float32),where=count>0)

def recover_native_probabilities(xyz, base, views, angles):
    """Original two-camera/spatial promotion; angles use the dataset convention."""
    xyz=np.asarray(xyz,np.float32);base=np.asarray(base,np.float32)
    views=np.asarray(views,np.float32);angles=np.asarray(angles,np.float32)
    if xyz.shape!=(len(base),3) or views.ndim!=2 or views.shape!=angles.shape or len(views)!=len(base):
        raise ValueError('Input coordinates, native field, camera probabilities and angles must align')
    if views.shape[1]<2:raise ValueError('At least two camera columns are required')
    valid=np.isfinite(views)&np.isfinite(angles)&(angles>=130)
    strong=valid&(views>=.25)
    cand=np.flatnonzero((strong.sum(axis=1)>=2)&(base>=.12)&(base<.25))
    supported=cand
    if len(cand):
        tree=cKDTree(xyz[base>=.12])
        local=tree.query_ball_point(xyz[cand],r=.012,return_length=True,workers=2)
        supported=cand[local>=6]
    new=base.copy()
    if len(supported):
        high2=np.partition(np.where(strong[supported],views[supported],-np.inf),-2,axis=1)[:,-2]
        target=np.minimum(np.minimum(.30,base[supported]+.06),.5*(base[supported]+high2))
        new[supported]=np.maximum(base[supported],target)
    return np.clip(new,0,1).astype(np.float32)
