import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Dict
import networkx as nx


class PrimatonNetwork:
    """Spatially-embedded geometric network according to Definition II.2"""

    def __init__(
        self,
        positions: np.ndarray = None,
        connectivity_radius: float = 0.1,
        n_nodes: int = 1000,
        domain_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):

        if positions is None:
            # Générer des positions aléatoires si non fournies
            np.random.seed(42)
            positions = np.random.uniform(0, 1, (n_nodes, 3)) * np.array(domain_size)

        self.positions = positions
        self.n_nodes = len(positions)
        self.r_c = connectivity_radius
        self.graph = nx.Graph()
        self.kd_tree = KDTree(positions)

        # Stockage des calendriers (quaternions)
        self.calendars = np.array(
            [self._random_unit_quaternion() for _ in range(self.n_nodes)]
        )

        # Données géométriques
        self.direction_vectors = {}
        self.rotation_operators = {}
        self.distances = {}

        self._build_network()

    def _random_unit_quaternion(self) -> np.ndarray:
        """Génère un quaternion unitaire aléatoire"""
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        return np.array([w, x, y, z])

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
        lambda_param = self.r_c / 3  # Interaction decay length

        for i, j in self.graph.edges():
            r_ij = self.positions[j] - self.positions[i]
            distance = np.linalg.norm(r_ij)

            # Direction unit vector
            n_ij = r_ij / distance if distance > 0 else np.array([1, 0, 0])
            self.direction_vectors[(i, j)] = n_ij
            self.direction_vectors[(j, i)] = -n_ij

            # Distance-dependent rotation angle (Eq. 9)
            theta_ij = np.pi * np.exp(-distance / lambda_param)

            # Rotation operator (Eq. 7)
            rotation_op = self._compute_rotation_operator(theta_ij, n_ij)
            self.rotation_operators[(i, j)] = rotation_op
            self.rotation_operators[(j, i)] = self._quaternion_conjugate(rotation_op)

            self.distances[(i, j)] = distance
            self.distances[(j, i)] = distance

    def _compute_rotation_operator(self, angle: float, axis: np.ndarray) -> np.ndarray:
        """Calcule l'opérateur de rotation quaternionique (Éq. 7)"""
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return np.array([w, xyz[0], xyz[1], xyz[2]])

    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Conjugué d'un quaternion"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node."""
        return list(self.graph.neighbors(node))

    def get_degree(self, node: int) -> int:
        """Get degree of a node."""
        return self.graph.degree[node]

    @property
    def average_degree(self) -> float:
        """Calculate average degree."""
        degrees = [d for _, d in self.graph.degree()]
        return np.mean(degrees) if degrees else 0

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        return nx.is_connected(self.graph)

    def get_laplacian_matrix(self) -> np.ndarray:
        """Get graph Laplacian matrix."""
        return nx.laplacian_matrix(self.graph).toarray()

    def get_geometric_data(self, i: int, j: int) -> dict:
        """Retourne les données géométriques pour l'arête (i,j)"""
        key = (i, j) if (i, j) in self.rotation_operators else (j, i)

        if key in self.rotation_operators:
            return {
                "distance": self.distances[key],
                "direction": self.direction_vectors[key],
                "angle": np.pi * np.exp(-self.distances[key] / (self.r_c / 3)),
                "rotation_operator": self.rotation_operators[key],
            }
        return {}

    def update_calendars(self, new_calendars: np.ndarray):
        """Met à jour les calendriers des nœuds"""
        self.calendars = new_calendars

    def get_nodes(self):
        """Retourne la liste des nœuds"""
        return list(self.graph.nodes())


# Fonctions utilitaires pour les quaternions
def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplication de quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugué d'un quaternion"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_norm(q: np.ndarray) -> float:
    """Norme d'un quaternion"""
    return np.sqrt(np.sum(q**2))
