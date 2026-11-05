"""Memory-bounded *exact-order* NetworkX-compatible complete Euclidean Kruskal MST.
Only replaces the dense O(n^2) Python NetworkX edge materialization; all LBC
and post-processing steps, edge weights, and stable lexicographic tie-order remain.
Validate whole prediction graph against original before adoption.
"""
import numpy as np
import networkx as nx
from pc_skeletor import LBC
from scipy.spatial.distance import cdist
from numba import njit

@njit(cache=True)
def _accept_sorted(order, start, stop, n):
    parent=np.arange(n,dtype=np.int64)
    rank=np.zeros(n,dtype=np.int64)
    accepted=np.empty(max(n-1,0),dtype=np.int64)
    found=0
    for k in order:
        a=start[k];b=stop[k]
        aa=a;bb=b
        while parent[aa]!=aa:
            parent[aa]=parent[parent[aa]]
            aa=parent[aa]
        while parent[bb]!=bb:
            parent[bb]=parent[parent[bb]]
            bb=parent[bb]
        if aa==bb:continue
        if rank[aa]<rank[bb]:
            parent[aa]=bb
        elif rank[aa]>rank[bb]:
            parent[bb]=aa
        else:
            parent[bb]=aa
            rank[aa]+=1
        accepted[found]=k
        found+=1
        if found==n-1:break
    return accepted[:found]

def contract_subcloud(subcloud,init_contraction=10.0,mst_backend="networkx"):
    if mst_backend!="networkx":
        raise ValueError("Validated exact path requires original NetworkX selection semantics")
    lbc=LBC(
        point_cloud=subcloud,
        down_sample=0.001,
        init_contraction=init_contraction,
        init_attraction=10.,
        max_attraction=2048,
        step_wise_contraction_amplification='auto',
        termination_ratio=0.003,
        max_iteration_steps=20,
        filter_nb_neighbors=20,
        filter_std_ratio=2.0
    )
    lbc.extract_skeleton()
    points=np.array(lbc.contracted_point_cloud.points)
    n=len(points)
    G=nx.Graph()
    if n==0:return G
    dist=cdist(points,points)
    ii,jj=np.triu_indices(n,k=1)
    weights=np.asarray(dist[ii,jj],dtype=np.float64)
    order=np.argsort(weights,kind="stable")
    accepted=_accept_sorted(order,ii,jj,n)
    # NetworkX original first adds all complete-graph nodes in increasing order,
    # then minimum_spanning_tree retains that node order before adding MST edges.
    G.add_nodes_from(np.arange(n,dtype=float))
    G.add_weighted_edges_from((float(ii[k]),float(jj[k]),float(weights[k])) for k in accepted)
    # byte-equivalent post processing of original contract_subcloud
    G.add_nodes_from(range(len(points)))
    nx.set_node_attributes(G,{i:points[i] for i in range(len(points))},"pos")
    degrees=np.array([val for node,val in G.degree()])
    furcations=np.where(2<degrees)[0]
    for furc_node in furcations:
        neighs=list(G.neighbors(furc_node))
        for i,neigh in enumerate(neighs):
            G.add_node(np.max(G.nodes)+1,pos=G.nodes[furc_node]["pos"])
            G.add_edge(np.max(G.nodes),neigh)
            G.remove_edge(furc_node,neigh)
        G.remove_node(furc_node)
    return G
