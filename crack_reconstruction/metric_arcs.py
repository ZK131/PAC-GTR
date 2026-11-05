import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import math
from scipy.sparse.csgraph import maximum_bipartite_matching


def sample(g,midpoint):
 consumed=set();chains=[];position=lambda n:tuple(g.nodes[n]['pos'])
 def take(s,t):
  route=[s];previous=s;current=t
  while True:
   consumed.add(frozenset((previous,current)));route.append(current)
   if g.degree(current)!=2 or current==s:break
   nextnode=next(x for x in g.neighbors(current) if x!=previous)
   if frozenset((current,nextnode)) in consumed:break
   previous,current=current,nextnode
  chains.append(route)
 for s in sorted(g,key=position):
  if g.degree(s)==2:continue
  for t in sorted(g.neighbors(s),key=position):
   if frozenset((s,t)) not in consumed:take(s,t)
 edge_order=list(g.edges) if midpoint else sorted(g.edges,key=lambda e:tuple(sorted((position(e[0]),position(e[1])))))
 for s,t in edge_order:
  if frozenset((s,t)) not in consumed:take(s,t)
 result=[]
 for chain in chains:
  points=np.array([g.nodes[n]['pos'] for n in chain],float)
  if tuple(points[-1])<tuple(points[0]):points=points[::-1]
  distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];total=distances[-1]
  if total<(1e-12 if midpoint else 1e-10):continue
  count=max(1,int(np.ceil(total/.0025)))
  query=(np.arange(count)+.5)*total/count if midpoint else np.linspace(0,total,max(2,count+1))
  result.append(np.column_stack([np.interp(query,distances,points[:,d]) for d in range(3)]))
 result=np.vstack(result) if result else np.empty((0,3))
 if not midpoint and len(result):result=result[np.unique(np.rint(result*1e8).astype(np.int64),axis=0,return_index=True)[1]]
 return result


def matched_f1(a,b,radius):
 if not len(a) or not len(b):return 0. if len(a)+len(b) else 1.
 matrix=cKDTree(b).sparse_distance_matrix(cKDTree(a),radius,output_type='coo_matrix');matrix.data=np.ones(len(matrix.data),np.int8)
 count=int(np.count_nonzero(maximum_bipartite_matching(matrix.tocsr(),perm_type='row')>=0));return 2*count/(len(a)+len(b))


def arc_f1(a,b):
 if not len(b):return 0.
 p=float(np.mean(cKDTree(a).query(b)[0]<=.005));r=float(np.mean(cKDTree(b).query(a)[0]<=.005));return 2*p*r/(p+r) if p+r else 0.


def endpoints(g):return np.array([g.nodes[n]['pos'] for n in g if g.degree(n)==1],float).reshape(-1,3)


def length(g):return math.fsum(np.linalg.norm(g.nodes[a]['pos']-g.nodes[b]['pos']) for a,b in g.edges)
