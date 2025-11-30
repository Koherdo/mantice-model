import pytest
import numpy as np
from src.primatron import PrimatonNetwork
from src.synchronization import QuaternionicSynchronization

<<<<<<< HEAD
=======

class TestPrimatonNetwork:
    def test_network_initialization(self):
        network = PrimatonNetwork(n_nodes=100, connectivity_radius=0.2)
        assert len(network.nodes) == 100
        assert network.average_degree > 0

    def test_quaternion_operations(self):
        # Test multiplication de quaternions
        q1 = np.array([1, 0, 0, 0])  # Quaternion unité
        q2 = np.array([0, 1, 0, 0])  # Quaternion i
        result = quaternion_multiply(q1, q2)
        assert np.allclose(result, np.array([0, 1, 0, 0]))

    def test_network_connectivity(self):
        network = PrimatonNetwork(n_nodes=50, connectivity_radius=0.3)
        for i in range(50):
            neighbors = network.get_neighbors(i)
            assert isinstance(neighbors, list)
>>>>>>> f6ff45180f478c35643408858a086b4daa2a8016


class TestSynchronization:
    def test_initialization(self):
        """Test de l'initialisation du système de synchronisation"""
        network = PrimatonNetwork(n_nodes=50)
        sync = QuaternionicSynchronization(network, coupling_strength=0.5)

        assert sync.sigma == 0.5
        assert sync.D == 0.01
        assert sync.intrinsic_frequencies.shape == (50, 4)

    def test_order_parameter(self):
        """Test du calcul du paramètre d'ordre"""
        network = PrimatonNetwork(n_nodes=10)
        sync = QuaternionicSynchronization(network)

<<<<<<< HEAD
        # Cas synchronisé
        sync_calendars = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
        R_sync = sync.compute_order_parameter(sync_calendars)
        assert abs(R_sync - 1.0) < 1e-10

        # Cas désynchronisé
=======
        # Calendriers parfaitement synchronisés
        identical_calendars = np.tile(np.array([1, 0, 0, 0]), (10, 1))
        R = sync.compute_order_parameter(identical_calendars)
        assert abs(R - 1.0) < 1e-10

        # Calendriers complètement désynchronisés
>>>>>>> f6ff45180f478c35643408858a086b4daa2a8016
        random_calendars = np.random.normal(0, 1, (10, 4))
        R_random = sync.compute_order_parameter(random_calendars)
        assert 0 <= R_random <= 1

<<<<<<< HEAD
    def test_derivative_computation(self):
        """Test du calcul de la dérivée"""
        network = PrimatonNetwork(n_nodes=5)
        sync = QuaternionicSynchronization(network, coupling_strength=0.1)

        calendars = network.calendars.copy()
        derivative = sync.compute_derivative(calendars, 0.0)

        assert derivative.shape == (5, 4)
        assert not np.allclose(derivative, 0)  # Ne devrait pas être nul

    def test_rk4_integration(self):
        """Test de l'intégration RK4"""
        network = PrimatonNetwork(n_nodes=10)
        sync = QuaternionicSynchronization(network, coupling_strength=0.1)

        initial_calendars = network.calendars.copy()
        new_calendars = sync.rk4_integration(initial_calendars, 0.01, 0.0)

        assert new_calendars.shape == (10, 4)
        # Les normes devraient rester proches de 1
        norms = [np.linalg.norm(q) for q in new_calendars]
        for norm in norms:
            assert 0.9 <= norm <= 1.1

    def test_short_evolution(self):
        """Test d'une évolution courte"""
        network = PrimatonNetwork(n_nodes=20)
        sync = QuaternionicSynchronization(network, coupling_strength=0.3)

        history = sync.evolve(timesteps=5, dt=0.01)

        assert "calendars" in history
        assert "order_parameter" in history
        assert "time" in history
        assert len(history["order_parameter"]) > 0

=======

class TestManticeDetection:
    def test_mantice_detector(self):
        detector = ManticeDetector(coherence_threshold=0.2)
        assert detector.epsilon == 0.2

    def test_union_find(self):
        detector = ManticeDetector()
        detector.initialize_union_find(5)

        detector.union(0, 1)
        detector.union(2, 3)

        assert detector.find(0) == detector.find(1)
        assert detector.find(2) == detector.find(3)
        assert detector.find(0) != detector.find(2)


def test_integration():
    """Test d'intégration complet"""
    network = PrimatonNetwork(n_nodes=100, connectivity_radius=0.15)
    sync = QuaternionicSynchronization(network, coupling_strength=0.6)

    # Simulation courte
    history = sync.evolve(timesteps=10, dt=0.01)

    assert "calendars" in history
    assert "order_parameter" in history
    assert len(history["order_parameter"]) > 0

>>>>>>> f6ff45180f478c35643408858a086b4daa2a8016

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
