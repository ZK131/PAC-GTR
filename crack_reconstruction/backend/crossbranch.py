import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def pos(g,n):return np.asarray(g.nodes[n]["pos"],float)


def extend(base,other,xyz,p):
 g=base.copy()
 ends=[n for n in base if base.degree[n]==1]
 if not ends:return g,{"qualified":0,"accepted":0,"added_nodes":0}
 ep=np.asarray([pos(base,n) for n in ends],float)
 tree=cKDTree(ep)
 all_nodes=list(base)
 base_pts=np.asarray([pos(base,n) for n in all_nodes],float)
 base_tree=cKDTree(base_pts)
 prob_tree=cKDTree(xyz)
 seen=[]
 cand=[]
 # Trace the other front-end's terminal back toward a V171 endpoint.
 for terminal in (n for n in other if other.degree[n]==1):
  if float(base_tree.query(pos(other,terminal))[0])<=.006:continue
  path=[terminal]
  prev=None
  cur=terminal
  dist=0.
  anchor=None
  for _ in range(16):
   d,idx=tree.query(pos(other,cur))
   if d<=.004:
    anchor=ends[int(idx)]
    break
   opts=[n for n in other.neighbors(cur) if n!=prev]
   if len(opts)!=1:break
   nxt=opts[0]
   step=float(np.linalg.norm(pos(other,nxt)-pos(other,cur)))
   if step<=1e-9 or step>.015 or dist+step>.080:break
   dist+=step
   path.append(nxt);prev,cur=cur,nxt
  if anchor is None:continue
  ordered=list(reversed(path))
  # Keep only points outside the V171 graph; no modification of existing nodes.
  novel=[n for n in ordered if float(base_tree.query(pos(other,n))[0])>.0045]
  if len(novel)<2:continue
  # Strictly a continuation of the V171 endpoint's local tangent.
  nb=next(iter(base.neighbors(anchor)))
  tangent=pos(base,anchor)-pos(base,nb)
  direction=pos(other,terminal)-pos(base,anchor)
  denom=float(np.linalg.norm(tangent)*np.linalg.norm(direction))
  if denom<1e-9:continue
  alignment=float(np.dot(tangent,direction)/denom)
  if alignment<.60:continue
  # Novel branch should connect locally, not jump across a crack gap.
  if float(np.linalg.norm(pos(other,novel[0])-pos(base,anchor)))>.014:continue
  q=[]
  for n in novel:
   ix=prob_tree.query_ball_point(pos(other,n),.004)
   q.append(float(np.max(p[ix])) if ix else 0.)
  if float(np.mean(q))<.25 or sum(v>=.25 for v in q)<2:continue
  pts=np.asarray([pos(other,n) for n in novel],float)
  if len(pts)>16:continue
  cand.append(dict(anchor=anchor,points=pts,mean_prob=float(np.mean(q)),alignment=alignment,
                   terminal=str(terminal),length=float(np.linalg.norm(pts[-1]-pos(base,anchor)))))
 cand.sort(key=lambda c:(-c["mean_prob"],-c["alignment"],str(c["anchor"])))
 accepted=0;added=0;next_id=max((int(n) for n in g if isinstance(n,(int,np.integer))),default=-1)+1
 used=set()
 for c in cand:
  if c["anchor"] in used:continue
  pts=c["points"]
  if seen and float(cKDTree(np.asarray(seen)).query(pts)[0].min())<.004:continue
  anchor=c["anchor"]
  last=anchor
  for pt in pts:
   node=next_id;next_id+=1
   g.add_node(node,pos=np.asarray(pt,float));g.add_edge(last,node)
   last=node;added+=1;seen.append(pt)
  used.add(anchor);accepted+=1
 assert set(base.nodes).issubset(g.nodes)
 assert set(map(frozenset,base.edges)).issubset(set(map(frozenset,g.edges)))
 assert nx.number_connected_components(g)==nx.number_connected_components(base)
 return g,dict(qualified=len(cand),accepted=accepted,added_nodes=added,
               mean_candidate_prob=float(np.mean([x["mean_prob"] for x in cand])) if cand else 0.,
               GT_opened=False)
