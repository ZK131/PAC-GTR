"""Equal-arc scorer used for the two Table6 domains; no implicit unit conversion."""
import numpy as np
from scipy.spatial import cKDTree
from .metric_arcs import sample,endpoints,matched_f1

def evaluate(reference,prediction,arc_tolerance,endpoint_tolerance):
    truth=sample(reference,False);pred=sample(prediction,False)
    if not len(truth):
        return {'positive':False,'predicted_nonempty':bool(len(pred))}
    if len(pred):
        a=cKDTree(truth).query(pred)[0];b=cKDTree(pred).query(truth)[0]
        p=float(np.mean(a<=arc_tolerance));r=float(np.mean(b<=arc_tolerance))
        cd=float((a.mean()+b.mean())/2)
    else:p=r=0.;cd=None
    return {'positive':True,'predicted_nonempty':bool(len(pred)),
            'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,
            'chamfer':cd,'endpoint_f1':matched_f1(endpoints(reference),endpoints(prediction),endpoint_tolerance)}
