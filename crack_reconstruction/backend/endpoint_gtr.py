import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
STEP_M=.005; MAX_STEPS=4; MIN_ADMITTED_STEPS=2; PROB_T=.25; PROB_RADIUS=.004; GRAPH_CLEARANCE=.004; TANGENT_MIN_BASE=.005; TANGENT_SPAN=.010


def extend(base,xyz,prob):
 new=base.copy()
 # All checks rely only on frozen 3D coordinates, probability and graph identity.
 cloud=cKDTree(xyz)
 graph_coords=np.asarray([np.asarray(base.nodes[n]["pos"],float) for n in base],float)
 graph_tree=cKDTree(graph_coords)
 fresh=[]
 next_id=max(int(n) for n in base.nodes if isinstance(n,(int,np.integer)))+1
 candidates=[]
 checked=0
 endpoints=[n for n in base.nodes if base.degree(n)==1]
 for node in endpoints:
  direct=next(iter(base.neighbors(node)))
  endpoint=np.asarray(base.nodes[node]["pos"],float)
  previous=node;current=direct
  distance=float(np.linalg.norm(endpoint-np.asarray(base.nodes[current]["pos"],float)))
  for _ in range(10):
   if distance>=TANGENT_SPAN or base.degree(current)!=2:break
   possible=[n for n in base.neighbors(current) if n!=previous]
   if len(possible)!=1:break
   following=possible[0]
   distance+=float(np.linalg.norm(np.asarray(base.nodes[following]["pos"])-np.asarray(base.nodes[current]["pos"])))
   previous,current=current,following
  far=np.asarray(base.nodes[current]["pos"],float)
  direction=endpoint-far
  norm=float(np.linalg.norm(direction))
  if norm<TANGENT_MIN_BASE:continue
  direction/=norm
  checked+=1
  good=[]
  for s in range(1,MAX_STEPS+1):
   point=endpoint+s*STEP_M*direction
   idx=cloud.query_ball_point(point,PROB_RADIUS)
   local_max=float(np.max(prob[idx])) if len(idx) else 0.0
   clearance=float(graph_tree.query(point)[0])
   if local_max<PROB_T or clearance<GRAPH_CLEARANCE:break
   good.append(point)
  if len(good)>=MIN_ADMITTED_STEPS:
   candidates.append(dict(node=node,points=good,prob_support_steps=len(good)))
 # Global collision arbitration uses source-only geometries, not GT.
 accepted=0;added=0
 for can in sorted(candidates,key=lambda z:(-z["prob_support_steps"],str(z["node"]))):
  future=np.asarray(can["points"],float)
  if fresh and float(cKDTree(np.asarray(fresh,float)).query(future)[0].min())<GRAPH_CLEARANCE:
   continue
  last=can["node"]
  for p in can["points"]:
   idx=next_id
   next_id+=1
   new.add_node(idx,pos=np.asarray(p,float))
   new.add_edge(last,idx)
   last=idx
   fresh.append(np.asarray(p,float))
   added+=1
  accepted+=1
 assert set(base.nodes).issubset(new.nodes)
 assert set(map(frozenset,base.edges)).issubset(set(map(frozenset,new.edges)))
 assert len(new)==len(base)+added
 assert nx.number_connected_components(new)==nx.number_connected_components(base)
 assert new.number_of_edges()==base.number_of_edges()+added
 return new,dict(source_endpoints=len(endpoints),checked_endpoints=checked,qualified_endpoint_candidates=len(candidates),accepted_noncolliding_endpoints=accepted,added_nodes=added,added_edges=added,recovered_length_mm=STEP_M*added*1000.,unmodified_original_node_positions=True,no_existing_nodes_or_edges_removed=True,topology_old_components_locked=True)
