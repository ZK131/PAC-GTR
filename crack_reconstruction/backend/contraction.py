import itertools
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from pc_skeletor import LBC
from scipy.spatial.distance import cdist


def contract_subcloud(subcloud, init_contraction=10., mst_backend="networkx"):
    # init Laplacian-based contraction
    lbc = LBC(point_cloud=subcloud,
              down_sample=0.001,
              init_contraction=init_contraction,
              init_attraction=10.,
              #max_contraction=32,
              max_attraction=2048,
              step_wise_contraction_amplification='auto',
              termination_ratio=0.003,
              max_iteration_steps=20,
              filter_nb_neighbors=20,
              filter_std_ratio=2.0,
              )
    lbc.extract_skeleton()

    points = np.array(lbc.contracted_point_cloud.points)

    # prepare fully connected graph with distances as edge weight and position as node attribute
    G = nx.Graph()
    distances = cdist(points, points)
    if mst_backend == "scipy":
        # The complete Euclidean graph is the same as the historical backend,
        # but SciPy avoids materialising O(n^2) Python NetworkX edge objects.
        # Dense zero-distance off-diagonal pairs must remain valid MST edges.
        from scipy.sparse.csgraph import minimum_spanning_tree
        distances[distances == 0] = np.finfo(np.float64).eps
        np.fill_diagonal(distances, 0)
        spanning = minimum_spanning_tree(distances).tocoo()
        G.add_weighted_edges_from(zip(spanning.row, spanning.col, spanning.data))
    elif mst_backend == "networkx":
        combinations = np.array(list(itertools.combinations(range(len(points)), 2)))
        entries = np.append(combinations, distances[combinations[:, 0], combinations[:, 1]][:, None], axis=-1)
        G.add_weighted_edges_from(entries)
        G = nx.minimum_spanning_tree(G)
    else:
        raise ValueError(f"Unknown mst_backend: {mst_backend}")
    G.add_nodes_from(range(len(points)))
    nx.set_node_attributes(G, {i: points[i] for i in range(len(points))}, "pos")
    degrees = np.array([val for (node, val) in G.degree()])
    furcations = np.where(2 < degrees)[0]

    # partition the tree
    for furc_node in furcations:
        neighs = list(G.neighbors(furc_node))
        for i, neigh in enumerate(neighs):
            G.add_node(np.max(G.nodes) + 1, pos=G.nodes[furc_node]["pos"])
            G.add_edge(np.max(G.nodes), neigh)
            G.remove_edge(furc_node, neigh)
        G.remove_node(furc_node)

    return G
