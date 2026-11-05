import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import math


def graph_same(g,h):
 return set(g.nodes)==set(h.nodes) and {frozenset(e) for e in g.edges}=={frozenset(e) for e in h.edges} and all(np.array_equal(np.asarray(g.nodes[n]["pos"]),np.asarray(h.nodes[n]["pos"])) for n in g)


def features(cs,after,dirs,graph_length):
 lens={};comp={}
 for j,nodes in enumerate(nx.connected_components(after)):
  nn=set(nodes);lens[j]=graph_length(after.subgraph(nn))
  for node in nn:comp[node]=j
 feat=[]
 for c in cs:
  a,b=c["a"],c["b"];pa=np.asarray(after.nodes[a]["pos"]);pb=np.asarray(after.nodes[b]["pos"])
  dist=float(np.linalg.norm(pb-pa));vec=(pb-pa)/max(dist,1e-12)
  left=float(np.dot(dirs[a],vec));right=float(np.dot(dirs[b],-vec))
  L=float(c["length"]);Q=float(c["quality"])
  la,lb=lens[comp[a]],lens[comp[b]]
  feat.append([math.log(1e-6+max(Q,0)),math.log(1e-5+L),math.log(1e-5+dist),
    left,right,math.sqrt(max(0,left)*max(0,right)),L/max(dist,1e-6),
    math.log(.01+la),math.log(.01+lb),max(0,L/(.03+.5*(la+lb))-1),
    float(len(c["support_ids"]))])
 return np.asarray(feat,dtype=np.float32).reshape(-1,11)
