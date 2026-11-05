import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import math
from dataclasses import dataclass, asdict
from .lifted_route import route, WeakAdjacency


@dataclass(frozen=True)
class Parameters:
    high:float=.25
    low:float=.15
    radius:float=.035
    minimum_lateral:float=.005
    direction_radius:float=.012
    neighbours:int=16
    maximum_empty_run:float=.010
    sample_spacing:float=.0025
    voxel:float=.0025
    weak_edge:float=.008
    maximum_gap:float=.040
    endpoint_radius:float=.006
    minimum_alignment:float=.5
    curvature_cost:float=.01
    maximum_detour:float=1.5


def graph_length(g):return math.fsum(float(np.linalg.norm(np.asarray(g.nodes[a]['pos'])-g.nodes[b]['pos'])) for a,b in g.edges)


def endpoint_direction(g,n,reach=.01):
    q=np.asarray(g.nodes[n]['pos']);cur=n;last=None;distance=0.
    while True:
        candidates=[a for a in g[cur] if a!=last]
        if len(candidates)!=1:break
        last,cur=cur,candidates[0];distance+=float(np.linalg.norm(np.asarray(g.nodes[cur]['pos'])-g.nodes[last]['pos']))
        if distance>=reach:break
    direction=q-np.asarray(g.nodes[cur]['pos']);return direction/max(np.linalg.norm(direction),1e-12)


def weak_network(xyz,normals,p,cfg):
    ids=np.flatnonzero(p>cfg.low)
    if not len(ids):return np.empty((0,3)),np.empty(0),[]
    keys=np.floor(xyz[ids]/cfg.voxel).astype(np.int64);_,inverse=np.unique(keys,axis=0,return_inverse=True);n=int(inverse.max())+1
    weights=p[ids];den=np.bincount(inverse,weights=weights,minlength=n)
    points=np.column_stack([np.bincount(inverse,weights=weights*xyz[ids,j],minlength=n)/den for j in range(3)])
    probability=np.bincount(inverse,weights=weights*p[ids],minlength=n)/den
    tree=cKDTree(points);distance,index=tree.query(points,k=min(13,n),distance_upper_bound=cfg.weak_edge,workers=1)
    if distance.ndim==1:distance=distance[:,None];index=index[:,None]
    adjacency=WeakAdjacency([[] for _ in range(n)])
    for i in range(n):
        for d,j in zip(distance[i],index[i]):
            if j<n and j!=i and np.isfinite(d):adjacency[i].append((int(j),float(d)))
    return points,probability,adjacency


def curved_wfr(base,xyz,normals,p,recovery_candidates,cfg=Parameters(),control='proposed'):
    from .evidence_refinement import ProbabilitySupport,EvidenceConfig,add_novel_supported_segments
    # Keep common weak-component recovery shape proposals. The changed stage
    # adds evidence-routed endpoint links, replacing straight greedy links.
    support=ProbabilitySupport(xyz,p,EvidenceConfig())
    result,recovery_audit=add_novel_supported_segments(base.copy(),recovery_candidates,support)
    points,prob,adjacency=weak_network(np.asarray(xyz),np.asarray(normals),np.asarray(p),cfg)
    tree=cKDTree(points) if len(points) else None
    ends=sorted(n for n,d in result.degree() if d==1);ep=np.array([result.nodes[n]['pos'] for n in ends])
    components={n:i for i,c in enumerate(nx.connected_components(result)) for n in c};directions={n:endpoint_direction(result,n) for n in ends}
    candidates=[]
    if len(ends)>1 and tree is not None:
        for i,j in sorted(cKDTree(ep).query_pairs(cfg.maximum_gap)):
            a,b=ends[i],ends[j]
            if components[a]==components[b]:continue
            candidate=route(ep[i],ep[j],directions[a],directions[b],points,prob,adjacency,tree,cfg)
            if candidate is None:continue
            f,run=support.edge_evidence(ep[i],ep[j])
            if control=='straight_only':
                if f<.8 or run>cfg.maximum_empty_run:continue
                candidate['points']=np.vstack([ep[i],ep[j]]);candidate['length']=float(np.linalg.norm(ep[j]-ep[i]))
            # Every routed segment must satisfy the common low-support rule.
            valid=True
            for x,y in zip(candidate['points'][:-1],candidate['points'][1:]):
                sf,sr=support.edge_evidence(x,y)
                if sf<.8 or sr>cfg.maximum_empty_run:valid=False;break
            if not valid:continue
            quality=float(np.mean(prob[candidate['support_ids']]))*np.linalg.norm(ep[j]-ep[i])/max(candidate['length'],1e-12)
            candidates.append({'a':a,'b':b,'quality':quality,**candidate})
    return result,{"candidates":candidates,"directions":directions,"weak_count":len(points),"recovery_info":recovery_audit}
    matching_graph=nx.Graph()
    for k,c in enumerate(candidates):matching_graph.add_edge(c['a'],c['b'],weight=c['quality'],proposal=k)
    selected=nx.max_weight_matching(matching_graph,maxcardinality=False,weight='weight')
    choice=sorted([matching_graph.edges[a,b]['proposal'] for a,b in selected],key=lambda k:(-candidates[k]['quality'],candidates[k]['a'],candidates[k]['b']))
    union=nx.utils.UnionFind(result);[union.union(a,b) for a,b in result.edges]
    cursor=max(result.nodes,default=-1)+1;accepted=[];cycle_rejections=0
    for k in choice:
        c=candidates[k];a,b=c['a'],c['b']
        if union[a]==union[b]:cycle_rejections+=1;continue
        ids=[a]
        for q in c['points'][1:-1]:result.add_node(cursor,pos=q);ids.append(cursor);cursor+=1
        ids.append(b)
        for left,right in zip(ids[:-1],ids[1:]):
            result.add_edge(left,right,weight=float(np.linalg.norm(np.asarray(result.nodes[left]['pos'])-result.nodes[right]['pos'])));union.union(left,right)
        accepted.append({'endpoints':[int(a),int(b)],'quality':c['quality'],'length_m':c['length'],'vertices':len(c['points'])})
    assert all(d<=2 for _,d in result.degree()),'Nonbranching contract violated; do not drop branches.'
    return result,{'recovery':recovery_audit,'weak_voxels':len(points),'path_proposals':len(candidates),'accepted_links':accepted,
         'cycle_rejections':cycle_rejections,'matching_scope':'maximum weight matching, followed by component-cycle filtering; not global forest optimum',
         'length_before_m':graph_length(base),'length_after_m':graph_length(result),'control':control,'config':asdict(cfg)}
