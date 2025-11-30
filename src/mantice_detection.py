import numpy as np
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import networkx as nx
from .primatron import PrimatonNetwork, quaternion_norm


class ManticeDetector:
    """Détecteur de Mantices selon l'Algorithme 1 et Définition II.8"""

    def __init__(self, coherence_threshold: float = 0.15, min_duration: int = 10):
        self.epsilon = coherence_threshold
        self.min_duration = min_duration
        self.union_find = None

    def compute_distance_matrix(self, calendars: np.ndarray) -> np.ndarray:
        """Calcule la matrice de distance quaternionique"""
        n_nodes = len(calendars)
        distance_matrix = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = self.quaternion_distance(calendars[i], calendars[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Distance entre deux quaternions"""
        diff = q1 - q2
        return np.sqrt(np.sum(diff**2))

    def initialize_union_find(self, n_nodes: int):
        """Initialise la structure Union-Find"""
        self.union_find = list(range(n_nodes))

    def find(self, x: int) -> int:
        """Trouve la racine avec compression de chemin"""
        if self.union_find[x] != x:
            self.union_find[x] = self.find(self.union_find[x])
        return self.union_find[x]

    def union(self, x: int, y: int):
        """Union de deux ensembles"""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.union_find[root_y] = root_x

    def detect_mantices(
        self, network: PrimatonNetwork, calendars: np.ndarray
    ) -> List[Set[int]]:
        """Détecte les Mantices selon l'Algorithme 1"""
        n_nodes = network.n_nodes

        # Étape 1: Matrice de distance
        distance_matrix = self.compute_distance_matrix(calendars)

        # Étape 2: Initialisation Union-Find
        self.initialize_union_find(n_nodes)

        # Étape 3: Union des nœuds cohérents
        edges = []
        for i in range(n_nodes):
            for j in network.get_neighbors(i):
                if j > i and distance_matrix[i, j] < self.epsilon:
                    edges.append((i, j, distance_matrix[i, j]))

        # Trier par distance croissante
        edges.sort(key=lambda x: x[2])

        # Appliquer les unions
        for i, j, dist in edges:
            if self.find(i) != self.find(j):
                self.union(i, j)

        # Étape 4: Extraction des composantes connexes
        components = defaultdict(set)
        for i in range(n_nodes):
            root = self.find(i)
            components[root].add(i)

        mantices = list(components.values())

        # Étape 5: Vérification de la connectivité spatiale
        verified_mantices = []
        for mantice in mantices:
            if self._check_spatial_connectivity(mantice, network):
                verified_mantices.append(mantice)

        return verified_mantices

    def _check_spatial_connectivity(
        self, mantice: Set[int], network: PrimatonNetwork
    ) -> bool:
        """Vérifie la connectivité spatiale du sous-graphe induit"""
        if len(mantice) <= 1:
            return True

        # Créer le sous-graphe
        subgraph_edges = []
        nodes_list = list(mantice)

        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                node_i, node_j = nodes_list[i], nodes_list[j]
                if node_j in network.get_neighbors(node_i):
                    subgraph_edges.append((node_i, node_j))

        # Vérifier la connectivité avec NetworkX
        G = nx.Graph()
        G.add_nodes_from(nodes_list)
        G.add_edges_from(subgraph_edges)

        return nx.is_connected(G)

    def compute_mantice_statistics(
        self, mantices: List[Set[int]], network: PrimatonNetwork
    ) -> Dict:
        """Calcule les statistiques des Mantices selon la Proposition II.11"""
        if not mantices:
            return {}

        sizes = [len(mantice) for mantice in mantices]

        statistics = {
            "number": len(mantices),
            "mean_size": np.mean(sizes),
            "size_distribution": sizes,
            "largest_mantice": max(sizes) if sizes else 0,
            "total_nodes_in_mantices": sum(sizes),
        }

        return statistics


def analyze_mantice_evolution(
    history: Dict, network: PrimatonNetwork, detector: ManticeDetector
) -> Dict:
    """Analyse l'évolution des Mantices dans le temps"""
    time_points = history["time"]
    calendars_history = history["calendars"]

    evolution_data = {
        "time": time_points,
        "mantice_count": [],
        "mean_mantice_size": [],
        "largest_mantice": [],
        "order_parameter": history["order_parameter"],
    }

    for i, calendars in enumerate(calendars_history):
        mantices = detector.detect_mantices(network, calendars)
        stats = detector.compute_mantice_statistics(mantices, network)

        evolution_data["mantice_count"].append(stats.get("number", 0))
        evolution_data["mean_mantice_size"].append(stats.get("mean_size", 0))
        evolution_data["largest_mantice"].append(stats.get("largest_mantice", 0))

    return evolution_data
