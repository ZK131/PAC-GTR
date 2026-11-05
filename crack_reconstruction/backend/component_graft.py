import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def graft(base,donor,xyz,p):
 g=base.copy()
 if not len(g):return g,dict(accepted=0,added_nodes=0,GT_opened=False)
 tree=cKDTree(np.asarray([base.nodes[n]["pos"] for n in base],float))
 cloud=cKDTree(xyz)
 next_id=max((int(n) for n in g if isinstance(n,(int,np.integer))),default=-1)+1
 accepted=0;added=0
 for comp in nx.connected_components(donor):
  nodes=list(comp)
  if len(nodes)<3:continue
  sub=donor.subgraph(nodes)
  length=sum(float(np.linalg.norm(np.asarray(donor.nodes[u]["pos"])-np.asarray(donor.nodes[v]["pos"]))) for u,v in sub.edges)
  if length<.015:continue
  pts=np.asarray([donor.nodes[n]["pos"] for n in nodes],float)
  d=tree.query(pts)[0]
  if float(np.mean(d>.005))<.75 or float(np.min(d))<=.003:continue
  q=[]
  for pt in pts:
   ix=cloud.query_ball_point(pt,.004)
   q.append(float(np.max(p[ix])) if ix else 0.)
  if float(np.mean(q))<.25:continue
  mapping={}
  for n in nodes:
   mapping[n]=next_id
   g.add_node(next_id,pos=np.asarray(donor.nodes[n]["pos"],float))
   next_id+=1;added+=1
  for u,v in sub.edges:g.add_edge(mapping[u],mapping[v])
  accepted+=1
 assert len(g)==len(base)+added
 assert nx.number_connected_components(g)==nx.number_connected_components(base)+accepted
 return g,dict(accepted=accepted,added_nodes=added,GT_opened=False)
