"""Executed dual-branch backend, relocated from the measured production runner.

Inputs are registered coordinates and aligned probability fields. Preparation of
the photographs and frozen weights is documented separately. No reference labels
or scene identity are accepted by reconstruction.
"""
import warnings
from types import SimpleNamespace
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from .backend.evidence_refinement import ProbabilitySupport, refine_centerlines
from .backend.evidence_refinement_v2 import refine_v2
from .backend import pac as v54, local_gtr as v51, endpoint_gtr as v171
from .backend import routes, route_features, route_selection, route_updates
from .backend import crossbranch as a2, component_graft as a7, integration as a34
v105=SimpleNamespace(b=routes, v100=route_features, wfr=route_updates)
v155=SimpleNamespace(b=routes, v100=route_features, wfr=route_updates, global_route=route_selection.global_route)
v215=SimpleNamespace(WFR=v155)

def reconstruct(xyz, normals, donor_probabilities, native_probabilities, ranker):
    """Reconstruct a graph using the frozen physical/normalized backend radii.

    Coordinates and radii must share a unit. For CrackEnsembles pass coordinates
    divided by the robust scene diagonal. The input adapter intentionally retains
    the inherited empty-seed rule. ranker must expose predict_proba(X).
    """
    xyz=np.asarray(xyz, dtype=float); norm=np.asarray(normals, dtype=float)
    prob=np.asarray(donor_probabilities, dtype=float)
    p2=np.asarray(native_probabilities, dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or norm.shape!=xyz.shape:
        raise ValueError("xyz and normals must have matching (N,3) shapes")
    if prob.shape!=(len(xyz),) or p2.shape!=prob.shape:
        raise ValueError("probability fields must be point-aligned (N,) arrays")
    if not all(np.all(np.isfinite(x)) for x in (xyz,norm,prob,p2)):
        raise ValueError("Inputs must be finite; apply visibility checks during mapping")
    if not len(xyz) or not (np.any(prob>=.25) and np.any(p2>=.25)):
        return nx.Graph()

    # Legacy donor branch: re-execute exact V54 no-support PAC, WFR, GTR, then V51.
    seeds=np.flatnonzero(prob>=v54.HITH);mask=prob>=v54.SUPPORT_THRESHOLD
    probtree=cKDTree(xyz[mask])
    groups,_=v54.cluster(xyz[seeds],prob[seeds],"v54_no_support_ablation",prob[mask],probtree)
    with warnings.catch_warnings():
     warnings.simplefilter("ignore")
     pac,_=v54.contract(xyz[seeds],groups)
    support=ProbabilitySupport(xyz,prob,v54.cfg())
    with warnings.catch_warnings():
     warnings.simplefilter("ignore")
     wfr,_=refine_centerlines(pac.copy(),support,candidates=None,prune=False,recover=True,link=True)
     full,_=refine_v2(wfr.copy(),support,wfr.copy(),mode="refine",trim_gap=.03,ridge_radius=.006)
    legacy_v55,_=v51.optimize(full.copy(),xyz,prob,"one_scale_4mm")
    # V105 source residual WFR (with frozen structure-specific ranker)
    generated,meta=v105.b.curved_wfr(wfr,xyz,norm,prob,nx.Graph(),cfg=v105.b.Parameters(maximum_gap=.065))
    assert v105.v100.graph_same(wfr,generated)
    cs=meta["candidates"];X=v105.v100.features(cs,generated,meta["directions"],v105.b.graph_length)
    with warnings.catch_warnings():
     warnings.simplefilter("ignore");q=ranker.predict_proba(X)[:,1] if len(cs) else np.empty(0)
    eligible=[k for k,c in enumerate(cs) if float(q[k])>=.70 and .04<float(np.linalg.norm(np.asarray(generated.nodes[c["a"]]["pos"])-np.asarray(generated.nodes[c["b"]]["pos"])))<=.065]
    order=sorted(eligible,key=lambda k:(-float(q[k]),-float(cs[k]["quality"]),k))
    uf=nx.utils.UnionFind(generated.nodes);used=set();chosen=[]
    for u,vv in generated.edges:uf.union(u,vv)
    for k in order:
     u,vv=cs[k]["a"],cs[k]["b"]
     if u in used or vv in used or uf[u]==uf[vv]:continue
     chosen.append(k);used.update((u,vv));uf.union(u,vv)
     if len(chosen)==2:break
    extended=v105.wfr.add_selected(generated,cs,chosen) if chosen else generated.copy()
    if chosen:
     with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      raw,_=refine_v2(extended.copy(),support,extended.copy(),mode="refine",trim_gap=.03,ridge_radius=.006)
     legacy_v105,_=v51.optimize(raw.copy(),xyz,prob,"one_scale_4mm")
    else:legacy_v105=legacy_v55.copy()
    # V155 uses V105 original WFR, not V105 extended WFR
    generated,meta=v155.b.curved_wfr(generated,xyz,norm,prob,nx.Graph(),cfg=v155.b.Parameters(maximum_gap=.065))
    assert v155.v100.graph_same(wfr,generated)
    cs=meta["candidates"];X=v155.v100.features(cs,generated,meta["directions"],v155.b.graph_length)
    with warnings.catch_warnings():
     warnings.simplefilter("ignore");q=ranker.predict_proba(X)[:,1] if len(cs) else np.empty(0)
    eligible,chosen=v155.global_route(cs,q,wfr)
    finalwfr=v155.wfr.add_selected(wfr,cs,chosen) if chosen else wfr.copy()
    if chosen:
     with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      ggtr,_=refine_v2(finalwfr.copy(),support,finalwfr.copy(),mode="refine",trim_gap=.03,ridge_radius=.006)
     donor_v155,_=v51.optimize(ggtr.copy(),xyz,prob,"one_scale_4mm")
    else:donor_v155=legacy_v55.copy()
    donor_v171,_=v171.extend(donor_v155,xyz,prob) if len(donor_v155) else (donor_v155.copy(),{})
    # Independently build native multi-view upgraded field branch from frozen 2D field.
    native_seeds=np.flatnonzero(p2>=v54.HITH);native_mask=p2>=v54.SUPPORT_THRESHOLD
    native_tree=cKDTree(xyz[native_mask])
    groups,_=v54.cluster(xyz[native_seeds],p2[native_seeds],"v54_no_support_ablation",p2[native_mask],native_tree)
    with warnings.catch_warnings():
     warnings.simplefilter("ignore")
     native_pac,_=v54.contract(xyz[native_seeds],groups)
    native_support=ProbabilitySupport(xyz,p2,v54.cfg())
    with warnings.catch_warnings():
     warnings.simplefilter("ignore")
     native_wfr,_=refine_centerlines(native_pac.copy(),native_support,candidates=None,prune=False,recover=True,link=True)
     native_gtr,_=refine_v2(native_wfr.copy(),native_support,native_wfr.copy(),mode="refine",trim_gap=.03,ridge_radius=.006)
    native_4mm,_=v51.optimize(native_gtr.copy(),xyz,p2,"one_scale_4mm")
    # V215->V226 new WFR/GTR
    generated,meta=v215.WFR.b.curved_wfr(native_wfr,xyz,norm,p2,nx.Graph(),cfg=v215.WFR.b.Parameters(maximum_gap=.065))
    assert v215.WFR.v100.graph_same(native_wfr,generated)
    cs=meta["candidates"];X=v215.WFR.v100.features(cs,generated,meta["directions"],v215.WFR.b.graph_length)
    with warnings.catch_warnings():
     warnings.simplefilter("ignore");q=ranker.predict_proba(X)[:,1] if len(cs) else np.empty(0)
    eligible,chosen=v215.WFR.global_route(cs,q,native_wfr)
    forward=[k for k in chosen if X[k,3]>=0 and X[k,4]>=0 and X[k,6]<=1.08]
    if forward:
     wf=v215.WFR.wfr.add_selected(native_wfr,cs,forward)
     with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      ggtr,_=refine_v2(wf.copy(),native_support,wf.copy(),mode="refine",trim_gap=.03,ridge_radius=.006)
     upgraded,_=v51.optimize(ggtr.copy(),xyz,p2,"one_scale_4mm")
    else:upgraded=native_4mm.copy()
    native_v226,_=v171.extend(upgraded,xyz,p2) if len(upgraded) else (upgraded.copy(),{})
    # Actual A034 donor/native fusion; do not reuse precomputed graph
    a5,_=a2.extend(native_v226,donor_v171,xyz,prob)
    a7out,_=a7.graft(a5,donor_v171,xyz,prob)
    final,_=a34.graft(a7out,donor_v171,xyz,prob,wfr_enabled=True,gtr_enabled=True) if len(donor_v171) else (nx.Graph(),{})
    return final
