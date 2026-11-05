import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from . import backbone


def wfr_endpoint_recover(out,base,xyz,prob,enabled=True):
    audit=dict(wfr_endpoint_candidates=0,wfr_endpoint_extensions=0,
        wfr_endpoint_nodes=0)
    if not enabled or not len(out) or not len(base):
        return out,audit
    base_ids=list(base)
    base_xyz=np.array([base.nodes[n]["pos"] for n in base_ids],float)
    base_tree=cKDTree(base_xyz)
    cloud=cKDTree(xyz)
    out_ids=list(out)
    out_xyz=np.array([out.nodes[n]["pos"] for n in out_ids],float)
    out_tree=cKDTree(out_xyz)
    # Existing endpoint tangent must align with the source-supported extension.
    endpoints=[n for n in out if out.degree(n)==1]
    next_id=max(int(n) for n in out if isinstance(n,(int,np.integer)))+1
    used=set()
    for endpoint in endpoints:
        if audit["wfr_endpoint_extensions"]>=12:
            break
        pe=np.asarray(out.nodes[endpoint]["pos"],float)
        nb=next(iter(out.neighbors(endpoint)))
        tangent=pe-np.asarray(out.nodes[nb]["pos"],float)
        tn=float(np.linalg.norm(tangent))
        if tn<1e-6:
            continue
        d,idx=base_tree.query(pe,k=min(3,len(base_ids)))
        choices=[]
        for dd,i in zip(np.atleast_1d(d),np.atleast_1d(idx)):
            if dd>.004:
                continue
            node=base_ids[int(i)]
            for cand in base.neighbors(node):
                pc=np.asarray(base.nodes[cand]["pos"],float)
                delta=pc-pe
                length=float(np.linalg.norm(delta))
                if not (.0035<=length<=.016):
                    continue
                if float(np.dot(tangent,delta)/(tn*length+1e-12))<.45:
                    continue
                if out_tree.query(pc)[0]<.0035:
                    continue
                near,ix=cloud.query(pc,k=5)
                if float(np.min(np.atleast_1d(near)))>.004:
                    continue
                evidence=float(np.max(prob[np.atleast_1d(ix)]))
                if evidence<.30:
                    continue
                q=np.array([pe+delta/3,pe+delta*2/3])
                _,qi=cloud.query(q,k=5)
                support=float(np.min(np.max(prob[qi],axis=1)))
                if support<.20:
                    continue
                audit["wfr_endpoint_candidates"]+=1
                choices.append((evidence+support,pc))
        if not choices:
            continue
        _,pc=max(choices,key=lambda x:x[0])
        key=tuple(np.round(pc,5))
        if key in used:
            continue
        used.add(key)
        out.add_node(next_id,pos=np.asarray(pc,float))
        out.add_edge(endpoint,next_id)
        next_id+=1
        audit["wfr_endpoint_extensions"]+=1
        audit["wfr_endpoint_nodes"]+=1
    return out,audit


def wfr_cloud_tangent_extend(out,xyz,prob,enabled=True,weak_support=False):
    # Source-only geometric continuation of existing endpoints; no new
    # components, no GT, fixed global thresholds, max six per case.
    audit=dict(wfr_cloud_candidates=0,wfr_cloud_extensions=0)
    if not enabled or not weak_support or len(out)<2:
        return out,audit
    cloud=cKDTree(xyz)
    nodes=list(out)
    nodepos=np.array([out.nodes[n]["pos"] for n in nodes],float)
    node_tree=cKDTree(nodepos)
    next_id=max(int(n) for n in out if isinstance(n,(int,np.integer)))+1
    for e in sorted((n for n in out if out.degree(n)==1),key=int):
        if audit["wfr_cloud_extensions"]>=6:break
        pe=np.asarray(out.nodes[e]["pos"],float)
        n=next(iter(out.neighbors(e)))
        tangent=pe-np.asarray(out.nodes[n]["pos"],float)
        tn=float(np.linalg.norm(tangent))
        if tn<.003:continue
        tangent=tangent/tn
        idx=cloud.query_ball_point(pe,r=.020)
        if not idx:continue
        cand=xyz[idx]-pe
        along=cand@tangent
        off=np.linalg.norm(cand-along[:,None]*tangent[None,:],axis=1)
        good=(along>=.006)&(along<=.020)&(off<=.0025)&(prob[idx]>=.50)
        if not np.any(good):continue
        ids=np.asarray(idx,dtype=int)[good]
        if not len(ids):continue
        distances=node_tree.query(xyz[ids])[0]
        ids=ids[distances>=.0045]
        if not len(ids):continue
        ranked=sorted(ids,key=lambda i:(-float(prob[i]),int(i)))
        for i in ranked[:10]:
            pc=np.asarray(xyz[i],float)
            q=np.vstack([pe+(pc-pe)/3,pe+(pc-pe)*2/3])
            d,ix=cloud.query(q,k=5)
            if np.any(np.min(d,axis=1)>.003):continue
            if float(np.min(np.max(prob[ix],axis=1)))<.35:continue
            audit["wfr_cloud_candidates"]+=1
            out.add_node(next_id,pos=pc)
            out.add_edge(e,next_id)
            next_id+=1
            audit["wfr_cloud_extensions"]+=1
            break
    return out,audit


def graft(base,donor,xyz,prob,wfr_enabled=True,gtr_enabled=True):
    assert backbone.graft is not None  # source hashes are recorded in SOURCE_PROVENANCE.json
    old=backbone
    # PAC source-expansion guard: when the recovered graph has >25% more
    # nodes than its PAC source, absent source nodes are not reliable
    # negative evidence. Preserve its native backbone; still run WFR.
    ratio=len(donor)/max(1,len(base))
    # A034: PAC native-backbone protection under uninformative 3D support.
    # Fixed .35 is the pre-existing A020 WFR target q75, not selected by GT.
    # Source-only confidence is measured on donor graph vertices.
    low_support=False
    if len(donor):
        donor_tree=cKDTree(xyz)
        donor_xyz=np.array([donor.nodes[n]["pos"] for n in donor],float)
        donor_local=prob[donor_tree.query(donor_xyz,k=5)[1]].max(axis=1)
        donor_q75=float(np.quantile(donor_local,.75))
        low_support=(donor_q75<.35)
    guarded=ratio>1.25
    # A034 source-only PAC integrity: low support protects the donor backbone,
    # but MUST NOT bypass the WFR and GTR upgrades in the same four-factor chain.
    active_base=donor if (guarded or low_support) else base
    # PAC and WFR are separated so only WFR receives local probability
    # calibration. This is a global GT-blind rule, not a per-method switch.
    out,audit=old.graft(active_base,donor,xyz,prob,wfr_enabled=False)
    prune_fraction=audit["removed_nodes"]/max(1,len(donor))
    budget_guard=prune_fraction>.08
    if budget_guard:
        active_base=donor
        out,audit=old.graft(active_base,donor,xyz,prob,wfr_enabled=False)
    cloud=cKDTree(xyz)
    donor_pos=np.array([donor.nodes[n]["pos"] for n in donor],float)
    local=prob[cloud.query(donor_pos,k=5)[1]].max(axis=1)
    med,q75=np.quantile(local,[.50,.75])
    # High-confidence expanded graph: avoid deleting a tiny number of
    # isolated native nodes when >10% source expansion is observed.
    native_guard=(ratio>1.10 and med>.80 and
        0<prune_fraction<.005)
    if native_guard:
        active_base=donor
        out,audit=old.graft(active_base,donor,xyz,prob,wfr_enabled=False)
    # Normalize WFR-only evidence on weak or mid-confidence frontends.
    scale=float(np.clip(.35/max(q75,.05),.4,3.0)) if med<.50 else 1.0
    wfr_prob=np.clip(prob*scale,0.,1.).astype(np.float32)
    out,ra=old.wfr_ridge_refine(out,xyz,wfr_prob,enabled=wfr_enabled)
    audit.update(ra)
    out,wa=wfr_endpoint_recover(out,active_base,xyz,wfr_prob,enabled=wfr_enabled)
    out,ca=wfr_cloud_tangent_extend(out,xyz,wfr_prob,enabled=wfr_enabled,
        weak_support=(med<.50))
    # A034 GTR: preserve missing native anchors only when source probability
    # >=0.5 and no upgraded node lies within fixed 3 mm physical tolerance.
    # Both thresholds are physical/source semantics, never fitted to GT.
    restored=[]
    orphan_rejected=0
    if gtr_enabled and len(donor) and len(out):
        upgraded_xyz=np.array([out.nodes[n]["pos"] for n in out],float)
        nearest=cKDTree(upgraded_xyz).query(donor_pos,k=1)[0]
        candidates={n for i,n in enumerate(donor)
                    if n not in out and nearest[i]>.003 and local[i]>=.5}
        # A034 GTR source-only component support: restore connected candidate
        # groups (>=2 donor nodes joined by native edges), or single nodes
        # attached to the existing graph. Reject unanchored singletons.
        # No GT, frontend identity or case-specific thresholds are consulted.
        for comp in nx.connected_components(donor.subgraph(candidates)):
            anchored=any(any(v in out for v in donor.neighbors(n)) for n in comp)
            coherent=(len(comp)>=2)
            if not (anchored or coherent):
                orphan_rejected+=len(comp)
                continue
            for n in sorted(comp):
                out.add_node(n,**donor.nodes[n])
                restored.append(n)
        restore_set=set(restored)
        for u,v,attrs in donor.edges(data=True):
            if (u in restore_set or v in restore_set) and u in out and v in out:
                out.add_edge(u,v,**attrs)
    return out,dict(**audit,**wa,**ca,
        gtr_high_support_anchor_restored=len(restored),
        gtr_orphan_anchor_rejected=orphan_rejected,pac_expansion_guard=guarded,
        pac_low_support_backbone_guard=bool(low_support),
        pac_source_expansion_ratio=round(ratio,6),
        pac_prune_budget_guard=budget_guard,
        pac_initial_prune_fraction=round(prune_fraction,6),
        pac_high_conf_native_guard=bool(native_guard),
        wfr_local_scale=round(scale,6),
        wfr_source_q50=round(float(med),6),
        wfr_source_q75=round(float(q75),6))
