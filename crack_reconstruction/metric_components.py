import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def canonical(g: nx.Graph) -> nx.Graph:
    # OBJ node/edge insertion order is intentionally discarded.
    items=sorted(g.nodes, key=lambda n:(tuple(float(x) for x in g.nodes[n]['pos']), int(n)))
    remap={old:i for i,old in enumerate(items)}
    out=nx.Graph()
    out.add_nodes_from(range(len(items)))
    nx.set_node_attributes(out,{remap[n]:np.asarray(g.nodes[n]['pos'],float) for n in items},'pos')
    out.add_edges_from(sorted((min(remap[a],remap[b]),max(remap[a],remap[b])) for a,b in g.edges if a!=b))
    return out


def length(g):
    return sum(float(np.linalg.norm(np.asarray(g.nodes[a]['pos'])-np.asarray(g.nodes[b]['pos']))) for a,b in g.edges)


def components(g):
    g=canonical(g)
    return [g.subgraph(c).copy() for c in nx.connected_components(g)]


def score_component(g, tree, prob):
    if len(g)==0:return 0.0
    pts=np.asarray([g.nodes[n]['pos'] for n in sorted(g.nodes)],float)
    _,idx=tree.query(pts,k=1)
    vals=np.nan_to_num(prob[np.asarray(idx)],nan=0.0)
    take=max(1,int(np.ceil(.20*len(vals))))
    return float(np.mean(np.sort(vals)[-take:]))


def ranked_components(g, tree, prob):
    cs=components(g)
    entries=[]
    for i,c in enumerate(cs):
        pos=np.asarray([c.nodes[n]['pos'] for n in c.nodes],float)
        centroid=tuple(np.mean(pos,axis=0)) if len(pos) else (0.,0.,0.)
        entries.append((score_component(c,tree,prob),length(c),centroid,i,c))
    entries.sort(key=lambda x:(-x[0],-x[1],x[2][0],x[2][1],x[2][2],x[3]))
    return [e[-1] for e in entries]
