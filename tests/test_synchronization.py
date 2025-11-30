import pytest
import numpy as np
from src.primatron import PrimatonNetwork, quaternion_multiply, quaternion_norm
from src.synchronization import QuaternionicSynchronization
from src.mantice_detection import ManticeDetector

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

class TestSynchronization:
    def test_sync_initialization(self):
        network = PrimatonNetwork(n_nodes=50)
        sync = QuaternionicSynchronization(network, coupling_strength=0.5)
        assert sync.sigma == 0.5
        assert sync.intrinsic_frequencies.shape == (50, 4)
    
    def test_order_parameter(self):
        network = PrimatonNetwork(n_nodes=10)
        sync = QuaternionicSynchronization(network)
        
        # Calendriers parfaitement synchronisés
        identical_calendars = np.tile(np.array([1, 0, 0, 0]), (10, 1))
        R = sync.compute_order_parameter(identical_calendars)
        assert abs(R - 1.0) < 1e-10
        
        # Calendriers complètement désynchronisés
        random_calendars = np.random.normal(0, 1, (10, 4))
        R = sync.compute_order_parameter(random_calendars)
        assert 0 <= R <= 1

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
    
    assert 'calendars' in history
    assert 'order_parameter' in history
    assert len(history['order_parameter']) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])