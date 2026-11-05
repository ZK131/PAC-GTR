import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import math


def global_route(candidates,p,base):
 comp={n:i for i,x in enumerate(nx.connected_components(base)) for n in x}
 elig=[k for k,c in enumerate(candidates) if float(p[k])>=.55 and
     .025<float(np.linalg.norm(np.asarray(base.nodes[c["a"]]["pos"],float)-
                              np.asarray(base.nodes[c["b"]]["pos"],float)))<=.065]
 if not elig:return elig,[]
 # V155 source-only component identity: WFR's job is to recover weak
 # short residuals, while joining two established curves is reserved for GTR.
 # Scale-free guard: if both original component arc lengths exceed the
 # geometric candidate chord, avoid merging these independently established
 # crack identities. No GT or per-case thresholds in inference.
 parts=list(nx.connected_components(base))
 cid={n:i for i,group in enumerate(parts) for n in group}
 clen=[]
 for group in parts:
  segment=base.subgraph(group)
  clen.append(sum(float(np.linalg.norm(np.asarray(base.nodes[a]["pos"])-np.asarray(base.nodes[b]["pos"]))) for a,b in segment.edges))
 safe=[]
 for k in elig:
  u,v=candidates[k]["a"],candidates[k]["b"]
  gap=float(np.linalg.norm(np.asarray(base.nodes[u]["pos"])-np.asarray(base.nodes[v]["pos"])))
  if min(clen[cid[u]],clen[cid[v]])<=gap+1e-12:safe.append(k)
 elig=safe
 if not elig:return elig,[]
 # GT-free mutual endpoint identity, fixed odds factor 2.
 endpoint_options={}
 for k in elig:
  u,v=candidates[k]["a"],candidates[k]["b"]
  if base.degree[u]!=1 or base.degree[v]!=1:continue
  endpoint_options.setdefault(u,[]).append(k)
  endpoint_options.setdefault(v,[]).append(k)
 def odds(q):
  q=float(np.clip(q,1e-10,1.-1e-10))
  return q/(1.-q)
 clear=[]
 for k in elig:
  u,v=candidates[k]["a"],candidates[k]["b"]
  if k not in endpoint_options.get(u,[]) or k not in endpoint_options.get(v,[]):continue
  value=odds(p[k])
  ambiguous=False
  for endpoint in (u,v):
   competition=max((odds(p[z]) for z in endpoint_options[endpoint] if z!=k),default=0.)
   if value<2.*competition:ambiguous=True;break
  if not ambiguous:clear.append(k)
 elig=clear
 if not elig:return elig,[]
 weights={i:math.log(max(1e-10,min(1-1e-10,float(p[i])))/(1-max(1e-10,min(1-1e-10,float(p[i]))))) for i in elig}
 order=sorted(elig,key=lambda i:(-weights[i],-float(candidates[i]["quality"]),i))
 best_score=0.;chosen=[]
 for i in order:
  u,v=candidates[i]["a"],candidates[i]["b"]
  if comp[u]==comp[v]:continue
  q=weights[i]
  if q>best_score+1e-14:chosen=[i];best_score=q
 # branch and bound: for each eligible link check non-reused endpoints and
 # component cycle (2 links same component-pair invalid)
 for i,first in enumerate(order):
  a,b_=candidates[first]["a"],candidates[first]["b"]
  a0,b0=comp[a],comp[b_]
  if a0==b0:continue
  for second in order[i+1:]:
   c,d=candidates[second]["a"],candidates[second]["b"]
   if len({a,b_,c,d})!=4:continue
   c0,d0=comp[c],comp[d]
   if c0==d0 or {a0,b0}=={c0,d0}:continue
   v=weights[first]+weights[second]
   if v>best_score+1e-14:
    best_score=v;chosen=[first,second]
 assert len(chosen)<=2 and len(chosen)<=len(elig)
 return elig,chosen
