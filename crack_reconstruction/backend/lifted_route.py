"""Sparse orientation-state path solver with post-search length feasibility."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class WeakAdjacency(list):pass


def lift(points,prob,adjacency,cfg):
    if hasattr(adjacency,'lifted'):return adjacency.lifted
    pairs=np.array([(a,b) for a,row in enumerate(adjacency) for b,_ in row],int).reshape(-1,2)
    outgoing=[[] for _ in points];incoming=[[] for _ in points]
    for i,(a,b) in enumerate(pairs):outgoing[a].append(i);incoming[b].append(i)
    if not len(pairs):return None
    delta=points[pairs[:,1]]-points[pairs[:,0]];distance=np.linalg.norm(delta,axis=1);direction=delta/np.maximum(distance[:,None],1e-12)
    costs=distance*(1.-np.log(np.maximum(np.minimum(prob[pairs[:,0]],prob[pairs[:,1]]),1e-6)))
    rows=[];cols=[];values=[]
    for i,(a,b) in enumerate(pairs):
        js=np.array(outgoing[b],int)
        if not len(js):continue
        cosine=direction[js]@direction[i];keep=(pairs[js,1]!=a)&(cosine>=0)
        js=js[keep];cc=cosine[keep]
        rows.extend([i]*len(js));cols.extend(js);values.extend(costs[js]+cfg.curvature_cost*(1-cc))
    graph=coo_matrix((values,(rows,cols)),shape=(len(pairs),len(pairs))).tocsr()
    result={'pairs':pairs,'direction':direction,'costs':costs,'graph':graph,'outgoing':outgoing,'incoming':incoming}
    adjacency.lifted=result
    return result


def route(a,b,ta,tb,points,prob,adjacency,tree,cfg):
    if not len(points):return None
    straight=float(np.linalg.norm(b-a))
    if straight<1e-9:return None
    limit=cfg.maximum_detour*straight+cfg.voxel
    starts=tree.query_ball_point(a,cfg.endpoint_radius);finishes=tree.query_ball_point(b,cfg.endpoint_radius)
    if not starts or not finishes:return None
    axis=(b-a)/straight;projection=(points-a)@axis
    near=np.linalg.norm(points-(a+np.clip(projection,0,straight)[:,None]*axis),axis=1)<=cfg.maximum_gap/2
    near &= np.linalg.norm(points-a,axis=1)+np.linalg.norm(points-b,axis=1)<=limit+1e-12
    data=lift(points,prob,adjacency,cfg)
    if data is None:return None
    pairs=data['pairs'];selected=np.flatnonzero(near[pairs[:,0]]&near[pairs[:,1]])
    if not len(selected):return None
    reverse=np.full(len(pairs),-1,int);reverse[selected]=np.arange(len(selected));n=len(selected);src=n;sink=n+1
    local=data['graph'][selected][:,selected].tocoo();rows=local.row.tolist();cols=local.col.tolist();values=local.data.tolist()
    for u in starts:
        initial=points[u]-a;length=float(np.linalg.norm(initial));direction=ta if length<=1e-9 else initial/length
        if length>1e-9 and ta@direction<cfg.minimum_alignment:continue
        ids=np.array(data['outgoing'][u],int)
        if not len(ids):continue
        cosine=data['direction'][ids]@direction;valid=(reverse[ids]>=0)&(cosine>=0)
        js=ids[valid];cc=cosine[valid]
        rows.extend([src]*len(js));cols.extend(reverse[js]);values.extend(length*(1-np.log(max(prob[u],1e-6)))+data['costs'][js]+cfg.curvature_cost*(1-cc))
    for v in finishes:
        tail=b-points[v];length=float(np.linalg.norm(tail));ids=np.array(data['incoming'][v],int)
        if not len(ids):continue
        if length<=1e-9:valid=(reverse[ids]>=0)&((-data['direction'][ids])@tb>=cfg.minimum_alignment)
        else:
            direction=tail/length
            if tb@(-direction)<cfg.minimum_alignment:continue
            valid=(reverse[ids]>=0)&(data['direction'][ids]@direction>=cfg.minimum_alignment)
        js=ids[valid];rows.extend(reverse[js]);cols.extend([sink]*len(js));values.extend([length]*len(js))
    graph=coo_matrix((values,(rows,cols)),shape=(n+2,n+2)).tocsr()
    distance,previous=dijkstra(graph,directed=True,indices=src,return_predecessors=True)
    if not np.isfinite(distance[sink]):return None
    path=[];cur=int(previous[sink])
    while cur!=src:
        if cur<0 or cur>=n:return None
        path.append(int(selected[cur]));cur=int(previous[cur])
    path.reverse()
    if not path:return None
    support=[int(pairs[path[0],0]),*[int(pairs[i,1]) for i in path]]
    q=np.vstack([a,points[support],b]);q=q[np.r_[True,np.linalg.norm(np.diff(q,axis=0),axis=1)>1e-9]]
    total=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
    if total>limit+1e-12:return None
    return {'points':q,'cost':float(distance[sink]),'length':total,'support_ids':support}
