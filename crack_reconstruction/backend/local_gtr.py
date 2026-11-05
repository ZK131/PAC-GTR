import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
RADIUS=.004; SHIFT_MAX=.0015; P_THRESHOLD=.15; ANCHOR=.25; CURVATURE=.25; ITERATIONS=2


def sections(g):
 return g.number_of_nodes(),g.number_of_edges(),nx.number_connected_components(g)


def optimize(g,xyz,prob,mode):
 h=g.copy()
 eligible=[n for n in g if g.degree(n)==2]
 original={n:np.asarray(g.nodes[n]["pos"],float).copy() for n in g}
 pos={n:x.copy() for n,x in original.items()}
 good=prob>P_THRESHOLD
 samples=xyz[good]
 q=prob[good]
 tree=cKDTree(samples) if len(samples) else None
 passes=ITERATIONS if mode=="selfbuilt_evidence_curvature_anchor" else 1
 changes=0;sumshift=0.
 for _ in range(passes):
  nextpos={n:v.copy() for n,v in pos.items()}
  for n in eligible:
   u,v=list(g.neighbors(n))
   p=pos[n];a=pos[u];b=pos[v]
   tangent=b-a;den=np.linalg.norm(tangent)
   if den<1e-10:continue
   tangent/=den
   if mode=="laplacian_025":
    delta=.25*((a+b)*.5-p)
   else:
    ids=tree.query_ball_point(p,RADIUS) if tree is not None else []
    if len(ids)<5:continue
    near=samples[ids];r=np.linalg.norm(near-p,axis=1)
    weights=q[ids]**2*np.exp(-.5*(r/(RADIUS*.6))**2)
    if weights.sum()<=1e-20:continue
    center=np.average(near,axis=0,weights=weights)
    if mode=="one_scale_4mm":delta=center-p
    elif mode=="selfbuilt_evidence_curvature_anchor":
     # Trust-region solution for quadratic evidence, curvature and original-position energies.
     e=(center-p);e-=np.dot(e,tangent)*tangent
     lap=(a+b)*.5-p;lap-=np.dot(lap,tangent)*tangent
     anchor=original[n]-p;anchor-=np.dot(anchor,tangent)*tangent
     curvature_cos=np.dot((a-p)/max(np.linalg.norm(a-p),1e-9),(b-p)/max(np.linalg.norm(b-p),1e-9))
     bend=np.sqrt(max(0.,2+2*curvature_cos))
     reliability=min(1.,len(ids)/8.)/(1.+bend)
     wd=max(.1,reliability)
     delta=(wd*e+CURVATURE*lap+ANCHOR*anchor)/(wd+CURVATURE+ANCHOR)
    else:raise ValueError(mode)
   delta-=np.dot(delta,tangent)*tangent
   if not np.isfinite(delta).all():continue
   new=p+delta
   total=new-original[n];length=np.linalg.norm(total)
   if length>SHIFT_MAX:new=original[n]+total*(SHIFT_MAX/length)
   nextpos[n]=new
  pos=nextpos
 for n in eligible:
  distance=np.linalg.norm(pos[n]-original[n])
  if distance>1e-12:
   h.nodes[n]["pos"]=pos[n];changes+=1;sumshift+=distance
 assert h.number_of_nodes()==g.number_of_nodes() and set(map(frozenset,h.edges))==set(map(frozenset,g.edges))
 assert all(np.array_equal(h.nodes[n]["pos"],original[n]) for n in g if g.degree(n)!=2)
 return h,dict(modified=changes,eligible=len(eligible),displacement_mm_sum=1000*sumshift,mean_modified_mm=1000*sumshift/max(1,changes),graph_topology_locked=True,endpoint_and_branch_locked=True)
