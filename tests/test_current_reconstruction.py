import unittest
import numpy as np
from crack_reconstruction.probability import visible_view_mean,recover_native_probabilities
from crack_reconstruction.ranker import LogisticRanker,fit_ranker,export_ranker

class CurrentReconstruction(unittest.TestCase):
    def test_visibility_excludes_nan_and_missing_views(self):
        p=np.array([[.8,.2,np.nan],[.7,.1,.5]],np.float32)
        valid=np.array([[1,1,1],[0,0,0]],bool)
        np.testing.assert_allclose(visible_view_mean(p,valid),[.5,0])
    def test_native_update_requires_camera_and_spatial_support(self):
        xyz=np.column_stack((np.arange(8)*.001,np.zeros((8,2))))
        p=np.full(8,.20,np.float32);p[0]=.9
        views=np.full((8,6),.1,np.float32);views[:,:2]=.8
        angles=np.full_like(views,150.)
        promoted=recover_native_probabilities(xyz,p,views,angles)
        self.assertEqual(promoted[0],p[0]);self.assertTrue(np.all(promoted[1:]>p[1:]))
        views[:,1]=.1
        np.testing.assert_array_equal(recover_native_probabilities(xyz,p,views,angles),p)
    def test_native_update_rejects_isolated_camera_agreement(self):
        xyz=np.eye(3,dtype=np.float32);p=np.full(3,.2,np.float32)
        views=np.full((3,6),.8,np.float32);angles=np.full_like(views,150.)
        np.testing.assert_array_equal(recover_native_probabilities(xyz,p,views,angles),p)
    def test_portable_ranker_matches_sklearn(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(40,11)).astype(np.float32)
        m=fit_ranker(x,(x[:,0]>0).astype(int));p=LogisticRanker.from_dict(export_ranker(m))
        np.testing.assert_allclose(m.predict_proba(x),p.predict_proba(x),rtol=1e-13,atol=1e-13)
    def test_empty_seed_gate(self):
        from crack_reconstruction.pipeline import reconstruct
        xyz=np.column_stack((np.arange(8)*.002,np.zeros((8,2))))
        normals=np.tile([0,0,1],(8,1))
        g=reconstruct(xyz,normals,np.zeros(8),np.ones(8),None)
        self.assertEqual(len(g),0)
    def test_chamfer_is_bidirectional_average(self):
        import networkx as nx
        from crack_reconstruction.metric_sampling import metrics
        from crack_reconstruction.metrics_transfer import evaluate
        a=np.array([[0.,0.,0.],[.01,0,0]])
        b=a+np.array([0,.02,0])
        self.assertAlmostEqual(metrics(a,b)['chamfer_mm'],20.)
        g=nx.Graph();g.add_node(0,pos=a[0]);g.add_node(1,pos=a[1]);g.add_edge(0,1)
        h=nx.Graph();h.add_node(0,pos=b[0]);h.add_node(1,pos=b[1]);h.add_edge(0,1)
        self.assertAlmostEqual(evaluate(g,h,.005,.04)['chamfer'],.02)
        self.assertIsNone(evaluate(g,nx.Graph(),.005,.04)['chamfer'])
    def test_image_gate_parameter_count_and_zero_head_initialization(self):
        import torch
        from crack_reconstruction.image.model_v7_context_cegate import ContextCEGate
        g=ContextCEGate();self.assertEqual(sum(p.numel() for p in g.parameters()),8691)
        p=torch.full((1,1,16,16),.4);q=torch.full_like(p,.6);rgb=torch.zeros((1,3,16,16))
        with torch.no_grad():out=torch.sigmoid(g(p,q,p,rgb))
        torch.testing.assert_close(out,q)

if __name__=='__main__':unittest.main()
