import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Dict
import networkx as nx
from .quaternions import Quaternion, create_rotation_quaternion


class Primaton:
    """Spatially-embedded geometric network."""

    def __init__(self, positions: np.ndarray, connectivity_radius: float):
        self.positions = positions
        self.n_nodes = len(positions)
        self.r_c = connectivity_radius
        self.graph = nx.Graph()
        self.kd_tree = KDTree(positions)
        self._build_network()

    def _build_network(self):
        """Build network using KD-tree for efficient neighbor search."""
        self.graph.add_nodes_from(range(self.n_nodes))

        # Find all pairs within connectivity radius
        pairs = self.kd_tree.query_pairs(self.r_c)

        for i, j in pairs:
            distance = np.linalg.norm(self.positions[i] - self.positions[j])
            if distance <= self.r_c:
                self.graph.add_edge(i, j, weight=distance)

        # Precompute geometric data
        self._precompute_geometric_data()

    def _precompute_geometric_data(self):
        """Precompute direction vectors and rotation operators."""
        self.direction_vectors = {}
        self.rotation_operators = {}
        self.distances = {}

        lambda_param = self.r_c / 3  # Interaction decay length

        for i, j in self.graph.edges():
            r_ij = self.positions[j] - self.positions[i]
            distance = np.linalg.norm(r_ij)

            # Direction unit vector
            n_ij = r_ij / distance if distance > 0 else np.array([1, 0, 0])
            self.direction_vectors[(i, j)] = n_ij
            self.direction_vectors[(j, i)] = -n_ij

            # Distance-dependent rotation angle
            theta_ij = np.pi * np.exp(-distance / lambda_param)

            # Rotation operator
            rotation_q = create_rotation_quaternion(theta_ij, n_ij)
            self.rotation_operators[(i, j)] = rotation_q
            self.rotation_operators[(j, i)] = rotation_q.inverse()

            self.distances[(i, j)] = distance
            self.distances[(j, i)] = distance

    def get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node."""
        return list(self.graph.neighbors(node))

    def get_degree(self, node: int) -> int:
        """Get degree of a node."""
        return self.graph.degree(node)

    def get_average_degree(self) -> float:
        """Calculate average degree."""
        degrees = [d for _, d in self.graph.degree()]
        return np.mean(degrees) if degrees else 0

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        return nx.is_connected(self.graph)

    def get_laplacian_matrix(self) -> np.ndarray:
        """Get graph Laplacian matrix."""
        return nx.laplacian_matrix(self.graph).toarray()