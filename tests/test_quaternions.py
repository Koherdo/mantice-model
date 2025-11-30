import pytest
import numpy as np
from src.primatron import quaternion_multiply, quaternion_conjugate, quaternion_norm


class TestQuaternionOperations:
    def test_quaternion_multiply(self):
        """Test de la multiplication de quaternions"""
        # Quaternion unité
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([1, 0, 0, 0])
        result = quaternion_multiply(q1, q2)
        expected = np.array([1, 0, 0, 0])
        assert np.allclose(result, expected)

        # Multiplication avec i
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        result = quaternion_multiply(q1, q2)
        expected = np.array([0, 1, 0, 0])
        assert np.allclose(result, expected)

        # i * i = -1
        q1 = np.array([0, 1, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        result = quaternion_multiply(q1, q2)
        expected = np.array([-1, 0, 0, 0])
        assert np.allclose(result, expected)

    def test_quaternion_conjugate(self):
        """Test du conjugué de quaternion"""
        q = np.array([1, 2, 3, 4])
        result = quaternion_conjugate(q)
        expected = np.array([1, -2, -3, -4])
        assert np.allclose(result, expected)

    def test_quaternion_norm(self):
        """Test de la norme de quaternion"""
        q = np.array([1, 0, 0, 0])
        assert abs(quaternion_norm(q) - 1.0) < 1e-10

        q = np.array([0, 1, 0, 0])
        assert abs(quaternion_norm(q) - 1.0) < 1e-10

        q = np.array([1, 1, 1, 1])
        expected_norm = np.sqrt(4)
        assert abs(quaternion_norm(q) - expected_norm) < 1e-10

    def test_quaternion_identity(self):
        """Test de l'identité multiplicative"""
        q = np.array([2, 3, 4, 5])
        identity = np.array([1, 0, 0, 0])

        result1 = quaternion_multiply(q, identity)
        result2 = quaternion_multiply(identity, q)

        assert np.allclose(result1, q)
        assert np.allclose(result2, q)

    def test_quaternion_associativity(self):
        """Test de l'associativité (approximative)"""
        q1 = np.array([1, 2, 3, 4])
        q2 = np.array([2, 3, 4, 5])
        q3 = np.array([3, 4, 5, 6])

        left = quaternion_multiply(quaternion_multiply(q1, q2), q3)
        right = quaternion_multiply(q1, quaternion_multiply(q2, q3))

        # Les quaternions sont associatifs
        assert np.allclose(left, right)


def test_synchronization_quaternion_functions():
    """Test des fonctions quaternioniques de synchronisation"""
    from src.synchronization import QuaternionicSynchronization
    from src.primatron import PrimatonNetwork

    network = PrimatonNetwork(n_nodes=10)
    sync = QuaternionicSynchronization(network)

    # Test de quaternion_sine avec petite valeur
    q_small = np.array([0.1, 0.0, 0.0, 0.0])
    result = sync.quaternion_sine(q_small)
    # Pour les petites valeurs, sin(q) ≈ q
    assert np.allclose(result, q_small, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
