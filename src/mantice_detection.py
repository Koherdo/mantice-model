import numpy as np
import networkx as nx
from typing import List, Set, Tuple
from .primatron import Primaton
from .synchronization import QuaternionicSynchronization
from collections import deque


class UnionFind:
    """Union-Find data structure with path compression and union-by-rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return

        # Union by rank
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1


class ManticeDetector:
    """Robust Mantice detection algorithm."""

    def __init__(self, primaton: Primaton, coherence_threshold: float = 0.15):
        self.primatron = primaton
        self.epsilon = coherence_threshold

    def find_bridges(self, graph: nx.Graph) -> List[Tuple[int, int]]:
        """Find bridge edges using Tarjan's algorithm."""
        if not graph.edges():
            return []

        # Convert to networkx graph for bridge detection
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(graph.edges())

        return list(nx.bridges(nx_graph))

    def detect_mantices(
        self, synchronization: QuaternionicSynchronization, time_window: int = 10
    ) -> List[Set[int]]:
        """Main Mantice detection algorithm."""

        # Step 1: Compute time-averaged distance matrix
        distance_matrix = self._compute_time_averaged_distances(
            synchronization, time_window
        )

        # Step 2: Union-Find clustering
        uf = self._union_find_clustering(distance_matrix)

        # Step 3: Extract connected components
        components = self._extract_components(uf)

        # Step 4: Spatial connectivity verification
        mantices = self._verify_spatial_connectivity(components)

        # Step 5: Temporal stability check
        mantices = self._temporal_stability_check(mantices, synchronization)

        return mantices

    def _compute_time_averaged_distances(
        self, synchronization: QuaternionicSynchronization, time_window: int
    ) -> np.ndarray:
        """Compute time-averaged quaternionic distance matrix."""
        n_nodes = self.primatron.n_nodes
        distance_matrix = np.zeros((n_nodes, n_nodes))

        # Sample multiple time points (simplified)
        for _ in range(time_window):
            coherence = synchronization.get_coherence_matrix()
            distance_matrix += coherence

        return distance_matrix / time_window

    def _union_find_clustering(self, distance_matrix: np.ndarray) -> UnionFind:
        """Perform Union-Find clustering based on coherence."""
        n_nodes = self.primatron.n_nodes
        uf = UnionFind(n_nodes)

        # Get edges sorted by coherence
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if distance_matrix[i, j] < self.epsilon:
                    edges.append((distance_matrix[i, j], i, j))

        # Sort by coherence (ascending - more coherent first)
        edges.sort()

        # Union nodes based on coherence
        for dist, i, j in edges:
            if distance_matrix[i, j] < self.epsilon:
                uf.union(i, j)

        return uf

    def _extract_components(self, uf: UnionFind) -> List[Set[int]]:
        """Extract connected components from Union-Find."""
        components = {}
        for i in range(self.primatron.n_nodes):
            root = uf.find(i)
            if root not in components:
                components[root] = set()
            components[root].add(i)

        return list(components.values())

    def _verify_spatial_connectivity(self, components: List[Set[int]]) -> List[Set[int]]:
        """Verify spatial connectivity of components."""
        mantices = []

        for component in components:
            if len(component) == 1:
                mantices.append(component)
                continue

            # Create subgraph for this component
            subgraph = self.primatron.graph.subgraph(component)

            if nx.is_connected(subgraph):
                mantices.append(component)
            else:
                # Find and cut weakest bridges
                connected_subcomponents = self._handle_disconnected_component(
                    subgraph, component
                )
                mantices.extend(connected_subcomponents)

        return mantices

    def _handle_disconnected_component(
        self, subgraph: nx.Graph, component: Set[int]
    ) -> List[Set[int]]:
        """Handle disconnected components by cutting weakest bridges."""
        bridges = self.find_bridges(subgraph)

        if not bridges:
            # No bridges found, return original connected components
            return [set(cc) for cc in nx.connected_components(subgraph)]

        # Find weakest bridge based on spatial distance
        weakest_bridge = min(
            bridges, key=lambda e: self.primatron.distances.get(e, float("inf"))
        )

        # Remove weakest bridge and get new components
        subgraph.remove_edge(*weakest_bridge)
        return [set(cc) for cc in nx.connected_components(subgraph)]

    def _temporal_stability_check(
        self, mantices: List[Set[int]], synchronization: QuaternionicSynchronization
    ) -> List[Set[int]]:
        """Check temporal stability of Mantices."""
        stable_mantices = []

        for mantice in mantices:
            if len(mantice) < 2:
                stable_mantices.append(mantice)
                continue

            # Compute mean calendar for this Mantice
            mean_calendar = self._compute_mean_calendar(mantice, synchronization)

            # Check stability of each node
            stable_nodes = set()
            for node in mantice:
                node_calendar = synchronization.calendars[node]
                distance = (node_calendar - mean_calendar).norm()

                if distance <= 1.5 * self.epsilon:
                    stable_nodes.add(node)

            if len(stable_nodes) >= 2:  # Minimum size for Mantice
                stable_mantices.append(stable_nodes)

        return stable_mantices

    def _compute_mean_calendar(
        self, mantice: Set[int], synchronization: QuaternionicSynchronization
    ) -> "Quaternion":
        """Compute mean quaternionic calendar for a Mantice."""
        from .quaternions import Quaternion

        sum_w, sum_x, sum_y, sum_z = 0, 0, 0, 0
        for node in mantice:
            cal = synchronization.calendars[node]
            sum_w += cal.w
            sum_x += cal.x
            sum_y += cal.y
            sum_z += cal.z

        n = len(mantice)
        return Quaternion(sum_w / n, sum_x / n, sum_y / n, sum_z / n).normalize()

    def get_mantice_statistics(self, mantices: List[Set[int]]) -> dict:
        """Compute statistics for detected Mantices."""
        if not mantices:
            return {}

        sizes = [len(m) for m in mantices]

        return {
            "number": len(mantices),
            "mean_size": np.mean(sizes),
            "size_std": np.std(sizes),
            "size_distribution": sizes,
            "largest_size": max(sizes),
            "smallest_size": min(sizes),
        }