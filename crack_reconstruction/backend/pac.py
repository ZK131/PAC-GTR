import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .contraction import contract_subcloud
from .evidence_refinement import EvidenceConfig
EPS=.035; HITH=.25; LOTH=.15; POWER=.05; CENTROID_MAX=EPS*3.0
SUPPORT_RADIUS=.005; SUPPORT_THRESHOLD=.15; SUPPORT_FRACTION_MIN=.25
MIN_LINEAR_POINTS=20; MIN_ANISOTROPY=.70
COS_CONTINUITY=float(np.cos(np.deg2rad(45.0))); TANGENT_K=12


def cfg():
 return EvidenceConfig(high_threshold=.25,low_threshold=.15,cluster_radius=.035,probability_power=.05,min_cluster_points=5,support_radius=.005,sample_spacing=.0025,max_unsupported_run=.010,novel_radius=.005,min_novel_length=.010,link_gap=.040,link_alignment=.866,link_support_fraction=.80,tangent_length=.010)


def cluster_shape(pts):
 n=len(pts)
 if n<3:return np.zeros(3),np.zeros(3),0.0
 center=pts.mean(axis=0);q=pts-center
 cov=q.T@q/max(n,1)
 eig,vec=np.linalg.eigh(cov);eig=np.maximum(eig,0)
 tangent=vec[:,-1]
 anis=float(np.clip((eig[-1]-eig[-2])/(eig[-1]+1e-12),0,1))
 return center,tangent,anis


def local_tangent(pts,at,k=TANGENT_K):
 if len(pts)<3:return np.zeros(3),0.0
 kk=min(k,len(pts))
 ix=cKDTree(pts).query(at,k=kk)[1]
 q=pts[np.asarray(ix).reshape(-1)]
 _,t,a=cluster_shape(q)
 return t,a


def merge_evidence(ida,idb,labels,x,p,bridge_probs,bridge_tree):
 ia=np.flatnonzero(labels==ida);ib=np.flatnonzero(labels==idb)
 pa=x[ia];pb=x[ib]
 ca,_,anis_a=cluster_shape(pa);cb,_,anis_b=cluster_shape(pb)
 # Find closest observed points, not centroid-to-centroid distance, for local bridge evidence.
 dist,nearest=cKDTree(pb).query(pa,k=1)
 ix=int(np.argmin(dist));jx=int(nearest[ix])
 xa=pa[ix];xb=pb[jx];gap=float(dist[ix]);vec=xb-xa
 if gap>1e-9:direction=vec/gap
 else:direction=np.zeros(3)
 ta,la=local_tangent(pa,xa);tb,lb=local_tangent(pb,xb)
 axis_agreement=float(abs(np.dot(ta,tb))) if np.linalg.norm(ta)>0 and np.linalg.norm(tb)>0 else 0.0
 end_align=float(min(abs(np.dot(ta,direction)),abs(np.dot(tb,direction)))) if gap>1e-9 else 1.0
 # Probe only the open interval. Original full point cloud supplies probability evidence.
 ns=min(32,max(1,int(np.ceil(gap/SUPPORT_RADIUS))-1))
 if gap<=SUPPORT_RADIUS:ns=1
 ts=np.linspace(0.0,1.0,ns+2)[1:-1]
 probes=xa[None,:]+ts[:,None]*vec[None,:]
 if len(bridge_probs):
  d_nn,ix_nn=bridge_tree.query(probes,k=1,distance_upper_bound=SUPPORT_RADIUS)
  supported=np.isfinite(d_nn) & (d_nn<=SUPPORT_RADIUS)
  safe_ix=np.clip(np.asarray(ix_nn,dtype=int),0,len(bridge_probs)-1)
  support_means=np.where(supported,bridge_probs[safe_ix],0.0)
  support_fraction=float(np.mean(supported))
  support_mean=float(np.mean(support_means))
 else:
  support_fraction=0.0
  support_mean=0.0
 elongated=(len(ia)>=MIN_LINEAR_POINTS and len(ib)>=MIN_LINEAR_POINTS and anis_a>=MIN_ANISOTROPY and anis_b>=MIN_ANISOTROPY)
 continuation=(gap<=SUPPORT_RADIUS or (axis_agreement>=COS_CONTINUITY and end_align>=COS_CONTINUITY))
 return dict(count_a=len(ia),count_b=len(ib),anis_a=anis_a,anis_b=anis_b,gap_m=gap,axis_agreement=axis_agreement,endpoint_alignment=end_align,support_fraction=support_fraction,support_mean=support_mean,elongated=bool(elongated),continuation=bool(continuation))


def cluster(x,p,mode,bridge_probs,bridge_tree):
 n=len(x);tree=cKDTree(x)
 pairs=tree.query_pairs(EPS,output_type="ndarray")
 if len(pairs):
  i,j=pairs[:,0],pairs[:,1]
  delta=x[j]-x[i];ds=np.linalg.norm(delta,axis=1)
  norm=ds/np.maximum((p[i]*p[j])**POWER,.01)
  ok=norm<=EPS
  ei,ej=i[ok],j[ok]
 else:
  ei=np.empty(0,dtype=int);ej=np.empty(0,dtype=int)
 mat=coo_matrix((np.ones(len(ei),dtype=np.uint8),(ei,ej)),shape=(n,n)).tocsr()
 count,labels=connected_components(mat,directed=False)
 labels=np.asarray(labels,dtype=np.int64)
 before=int(count);unique=sorted(int(z) for z in np.unique(labels))
 centroids={z:x[labels==z].mean(axis=0) for z in unique}
 merge_count=0;veto_count=0;candidate_count=0;evidence_calls=0;ev_log=[]
 # Preserve V53's exact greedy centroid merge order; the only intervention is a veto on high-risk merges.
 if mode!="no_merge_control":
  changed=True
  while changed:
   changed=False
   for i in range(len(unique)):
    for j in range(i+1,len(unique)):
     a,b=unique[i],unique[j]
     if np.linalg.norm(centroids[a]-centroids[b])>=CENTROID_MAX:continue
     candidate_count+=1
     if float(p[labels==a].mean())<=.1 or float(p[labels==b].mean())<=.1:continue
     ev=None;veto=False
     if mode!="legacy_centroid_merge_rebuilt":
      ev=merge_evidence(a,b,labels,x,p,bridge_probs,bridge_tree);evidence_calls+=1
      low_support=ev["support_fraction"]<SUPPORT_FRACTION_MIN
      bad_geometry=not ev["continuation"]
      if mode=="v54_evidence_gated_merge":
       veto=ev["elongated"] and low_support and bad_geometry
      elif mode=="v54_no_support_ablation":
       veto=ev["elongated"] and bad_geometry
      elif mode=="v54_no_geometry_ablation":
       veto=ev["elongated"] and low_support
      else:raise ValueError("unknown merge mode "+mode)
      if veto:
       veto_count+=1
       if len(ev_log)<500:ev_log.append(dict(a=a,b=b,decision="veto",**ev))
       continue
      if len(ev_log)<500:ev_log.append(dict(a=a,b=b,decision="merge",**ev))
     labels[labels==b]=a
     centroids[a]=x[labels==a].mean(axis=0)
     del centroids[b];unique.pop(j)
     merge_count+=1;changed=True;break
    if changed:break
 ids=np.unique(labels)
 groups=[np.flatnonzero(labels==s) for s in ids if np.count_nonzero(labels==s)>=5]
 audit=dict(mode=mode,n_candidates=n,n_pairs_within_eps=len(pairs),eligible_probability_edges=len(ei),components_before_merge=before,components_after_merge=len(ids),merge_candidates=candidate_count,accepted_merges=merge_count,vetoed_merges=veto_count,evidence_evaluations=evidence_calls,contract_clusters=len(groups),cluster_point_total=sum(map(len,groups)),evidence_examples=ev_log)
 return groups,audit


def contract(x,groups):
 result=nx.Graph();fail=[]
 for idx,ids in enumerate(groups):
  cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x[ids]))
  try:
   gr=contract_subcloud(cloud,init_contraction=10.0)
   gr=nx.relabel_nodes(gr,{v:k+len(result) for k,v in enumerate(gr.nodes)},copy=True)
   result=nx.compose(result,gr)
  except Exception as ex:fail.append(dict(cluster=idx,points=len(ids),exception=repr(ex)))
 return result,fail
