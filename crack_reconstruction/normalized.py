"""CrackEnsembles boundary adapter, retaining frozen small-cloud handling.

Run each reconstruction in its own process. The temporary dependency overrides
are process-local and are restored on exit; this function is not thread-safe.
"""
def reconstruct_normalized(xyz,normals,omni_probabilities,learned_dutr_probabilities,ranker):
    import robust_laplacian
    from . import pipeline
    from .backend.normalized_contraction import contract_subcloud
    previous=robust_laplacian.point_cloud_laplacian
    previous_contract=pipeline.v54.contract_subcloud
    def small_cloud(points,mollify_factor=1e-5,n_neighbors=30):
        try:return previous(points,mollify_factor=mollify_factor,n_neighbors=n_neighbors)
        except RuntimeError as exc:
            if 'k+1' not in str(exc) or len(points)<4:raise
            return previous(points,mollify_factor=mollify_factor,n_neighbors=min(n_neighbors,len(points)-1))
    try:
        robust_laplacian.point_cloud_laplacian=small_cloud
        pipeline.v54.contract_subcloud=contract_subcloud
        return pipeline.reconstruct(xyz,normals,omni_probabilities,learned_dutr_probabilities,ranker)
    finally:
        robust_laplacian.point_cloud_laplacian=previous
        pipeline.v54.contract_subcloud=previous_contract
