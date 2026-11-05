import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def wfr_ridge_refine(out,xyz,prob,enabled=True):
 # Adaptive-WFR-only, source-evidence-guided densification of existing edges.
 # Never create a new component or a new bridge; keep endpoints and topology.
 audit=dict(wfr_long_edges=0,wfr_supported_edges=0,wfr_ridge_nodes=0)
 if not enabled or not len(out):return out,audit
 cloud=cKDTree(xyz)
 ids=list(out)
 node_tree=cKDTree(np.array([out.nodes[n]["pos"] for n in ids],float))
 edges=[]
 for u,v in out.edges:
  pu=np.asarray(out.nodes[u]["pos"],float);pv=np.asarray(out.nodes[v]["pos"],float)
  length=float(np.linalg.norm(pv-pu))
  if .012<=length<=.045:edges.append((length,u,v))
 for length,u,v in sorted(edges,reverse=True):
  audit["wfr_long_edges"]+=1
  pu=np.asarray(out.nodes[u]["pos"],float);pv=np.asarray(out.nodes[v]["pos"],float)
  mid=(pu+pv)*.5
  dist,near=cloud.query(mid,k=8)
  eligible=[int(i) for d,i in zip(np.atleast_1d(dist),np.atleast_1d(near)) if d<=.003 and prob[int(i)]>=.35]
  if not eligible:continue
  best=max(eligible,key=lambda i:float(prob[i]))
  ridge=xyz[best]
  vec=pv-pu
  alpha=float(np.dot(ridge-pu,vec)/(length*length))
  if not (.25<=alpha<=.75):continue
  off=float(np.linalg.norm(ridge-(pu+alpha*vec)))
  if off>.003:continue
  if node_tree.query(ridge)[0]<.0045:continue
  q1=pu+vec/3.;q2=pu+vec*2./3.
  _,qi=cloud.query(np.vstack([q1,q2]),k=5)
  if float(np.min(np.max(prob[qi],axis=1)))<.25:continue
  audit["wfr_supported_edges"]+=1
  if not out.has_edge(u,v):continue
  nid=max(int(n) for n in out if isinstance(n,(int,np.integer)))+1
  out.remove_edge(u,v)
  out.add_node(nid,pos=np.asarray(ridge,float))
  out.add_edge(u,nid);out.add_edge(nid,v)
  audit["wfr_ridge_nodes"]+=1
  if audit["wfr_ridge_nodes"]>=16:break
 return out,audit


def graft(base,donor,xyz,prob,wfr_enabled=True):
 # A011 factor-confidence pruning plus A007 source-supported short arc
 # recovery. All thresholds fixed globally, no GT in graph generation.
 out=donor.copy()
 tree=cKDTree(np.array([base.nodes[n]["pos"] for n in base],float))
 cloud=cKDTree(xyz)
 removed_components=0;removed_nodes=0;added_arcs=0;added_nodes=0
 for comp in nx.connected_components(donor):
  ids=list(comp)
  if len(ids)<3 or len(ids)>30:continue
  sub=donor.subgraph(ids)
  length=sum(float(np.linalg.norm(np.asarray(donor.nodes[u]["pos"])-np.asarray(donor.nodes[v]["pos"]))) for u,v in sub.edges)
  if length>.060:continue
  pos=np.array([donor.nodes[n]["pos"] for n in ids],float)
  dd=tree.query(pos)[0]
  if float(np.mean(dd>.004))<.50:continue
  pp=np.max(prob[cloud.query(pos,k=5)[1]],axis=1)
  if float(np.mean(pp))>=.45:continue
  removed_components+=1;removed_nodes+=len(ids)
  out.remove_nodes_from(ids)
 if not len(out):
  return out,dict(wfr_long_edges=0,wfr_supported_edges=0,wfr_ridge_nodes=0,removed_components=removed_components,removed_nodes=removed_nodes,added_arcs=0,added_nodes=0,source_only=True,empty_all_pruned_boundary=True)
 out_tree=cKDTree(np.array([out.nodes[n]["pos"] for n in out],float))
 next_id=max(int(n) for n in out if isinstance(n,(int,np.integer)))+1
 for comp in nx.connected_components(base):
  ids=list(comp)
  if len(ids)<2:continue
  pos=np.array([base.nodes[n]["pos"] for n in ids],float)
  dd=out_tree.query(pos)[0]
  pp=np.max(prob[cloud.query(pos,k=5)[1]],axis=1)
  eligible={n for n,d,p in zip(ids,dd,pp) if d>=.003 and p>=.25}
  if len(eligible)<2:continue
  for arc in nx.connected_components(base.subgraph(eligible)):
   ids=list(arc)
   if len(ids)<2:continue
   sub=base.subgraph(ids)
   length=sum(float(np.linalg.norm(np.asarray(base.nodes[u]["pos"])-np.asarray(base.nodes[v]["pos"]))) for u,v in sub.edges)
   if length<.005:continue
   pos=np.array([base.nodes[n]["pos"] for n in ids],float)
   if float(np.median(out_tree.query(pos)[0]))>.035:continue
   if float(np.mean(np.max(prob[cloud.query(pos,k=5)[1]],axis=1)))<.30:continue
   mapping={}
   for n in ids:
    mapping[n]=next_id;out.add_node(next_id,pos=np.asarray(base.nodes[n]["pos"],float))
    next_id+=1;added_nodes+=1
   for u,v in sub.edges:out.add_edge(mapping[u],mapping[v])
   added_arcs+=1
 out,wfr_audit=wfr_ridge_refine(out,xyz,prob,wfr_enabled)
 return out,dict(**wfr_audit,removed_components=removed_components,removed_nodes=removed_nodes,
  added_arcs=added_arcs,added_nodes=added_nodes,source_only=True,
  backbone="V171",upgrade_source="A007 full four-factor")
