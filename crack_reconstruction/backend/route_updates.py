import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def add_selected(base,candidates,indices):
 g=base.copy();nextid=max(g.nodes,default=-1)+1;used=set();uf=nx.utils.UnionFind(g.nodes)
 for u,v in g.edges:uf.union(u,v)
 for rawk in indices:
  c=candidates[int(rawk)];a,bn=int(c["a"]),int(c["b"])
  if a not in g or bn not in g or a in used or bn in used or uf[a]==uf[bn]:raise AssertionError(("invalid selected endpoints",a,bn))
  pts=np.asarray(c["points"],float);pa=np.asarray(g.nodes[a]["pos"],float);pb=np.asarray(g.nodes[bn]["pos"],float)
  direct=np.linalg.norm(pts[0]-pa)+np.linalg.norm(pts[-1]-pb);reverse=np.linalg.norm(pts[0]-pb)+np.linalg.norm(pts[-1]-pa)
  if reverse<direct:a,bn=bn,a;pa,pb=pb,pa;pts=pts[::-1]
  if max(np.linalg.norm(pts[0]-pa),np.linalg.norm(pts[-1]-pb))>1e-5:raise AssertionError("candidate route endpoint mismatch")
  prev=a;prevpos=pa.copy()
  for q in pts[1:-1]:
   q=np.asarray(q,float)
   if np.linalg.norm(q-prevpos)<1e-10 or np.linalg.norm(q-pb)<1e-10:continue
   nid=nextid;nextid+=1;g.add_node(nid,pos=q.copy());g.add_edge(prev,nid,weight=float(np.linalg.norm(q-prevpos)));prev=nid;prevpos=q
  if prev!=bn:g.add_edge(prev,bn,weight=float(np.linalg.norm(pb-prevpos)))
  used.update((a,bn));uf.union(a,bn)
 return g
