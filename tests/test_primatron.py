import pytest
import numpy as np
from src.primatron import PrimatonNetwork, quaternion_multiply, quaternion_norm


class TestPrimatonNetwork:
    def test_network_initialization(self):
        """Test de l'initialisation du réseau Primaton"""
        network = PrimatonNetwork(n_nodes=100, connectivity_radius=0.2)
        assert network.n_nodes == 100
        assert network.average_degree > 0
        assert len(list(network.graph.nodes())) == 100

    def test_quaternion_operations(self):
        """Test des opérations de quaternions"""
        # Test multiplication de quaternions
        q1 = np.array([1, 0, 0, 0])  # Quaternion unité
        q2 = np.array([0, 1, 0, 0])  # Quaternion i
        result = quaternion_multiply(q1, q2)
        expected = np.array([0, 1, 0, 0])
        assert np.allclose(result, expected)

    def test_network_connectivity(self):
        """Test de la connectivité du réseau"""
        network = PrimatonNetwork(n_nodes=50, connectivity_radius=0.3)
        for i in range(50):
            neighbors = network.get_neighbors(i)
            assert isinstance(neighbors, list)

    def test_geometric_data(self):
        """Test des données géométriques"""
        network = PrimatonNetwork(n_nodes=20, connectivity_radius=0.2)
        # Vérifier qu'il y a des données géométriques pour certaines arêtes
        has_geometric_data = False
        for i in range(20):
            neighbors = network.get_neighbors(i)
            for j in neighbors:
                data = network.get_geometric_data(i, j)
                if data:
                    has_geometric_data = True
                    break
            if has_geometric_data:
                break
        assert has_geometric_data

    def test_node_positions(self):
        """Test des positions des nœuds"""
        network = PrimatonNetwork(n_nodes=30, connectivity_radius=0.1)
        assert network.positions.shape == (30, 3)
        # Vérifier que les positions sont dans le domaine [0,1]
        assert np.all(network.positions >= 0)
        assert np.all(network.positions <= 1)


class TestSynchronization:
    def test_sync_initialization(self):
        """Test de l'initialisation de la synchronisation"""
        from src.synchronization import QuaternionicSynchronization

        network = PrimatonNetwork(n_nodes=50)
        sync = QuaternionicSynchronization(network, coupling_strength=0.5)
        assert sync.sigma == 0.5
        assert sync.intrinsic_frequencies.shape == (50, 4)

    def test_order_parameter(self):
        """Test du paramètre d'ordre"""
        from src.synchronization import QuaternionicSynchronization

        network = PrimatonNetwork(n_nodes=10)
        sync = QuaternionicSynchronization(network)

        # Calendriers parfaitement synchronisés
        identical_calendars = np.tile(np.array([1, 0, 0, 0]), (10, 1))
        R = sync.compute_order_parameter(identical_calendars)
        assert abs(R - 1.0) < 1e-10

        # Calendriers aléatoires
        random_calendars = np.random.normal(0, 1, (10, 4))
        R = sync.compute_order_parameter(random_calendars)
        assert 0 <= R <= 1


class TestManticeDetection:
    def test_mantice_detector(self):
        """Test du détecteur de Mantices"""
        from src.mantice_detection import ManticeDetector

        detector = ManticeDetector(coherence_threshold=0.2)
        assert detector.epsilon == 0.2

    def test_union_find(self):
        """Test de la structure Union-Find"""
        from src.mantice_detection import ManticeDetector

        detector = ManticeDetector()
        detector.initialize_union_find(5)

        detector.union(0, 1)
        detector.union(2, 3)

        assert detector.find(0) == detector.find(1)
        assert detector.find(2) == detector.find(3)
        assert detector.find(0) != detector.find(2)


def test_integration():
    """Test d'intégration complet"""
    from src.primatron import PrimatonNetwork
    from src.synchronization import QuaternionicSynchronization

    network = PrimatonNetwork(n_nodes=100, connectivity_radius=0.15)
    sync = QuaternionicSynchronization(network, coupling_strength=0.6)

    # Simulation courte
    history = sync.evolve(timesteps=10, dt=0.01)

    assert "calendars" in history
    assert "order_parameter" in history
    assert len(history["order_parameter"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
