import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.primatron import Primaton

class TestPrimaton:
    def test_primatron_creation(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        primaton = Primaton(positions, 1.5)
        assert primaton.n_nodes == 3
        assert primaton.get_average_degree() > 0
    
    def test_primatron_connectivity(self):
        positions = np.array([[0, 0, 0], [0.5, 0, 0], [2, 0, 0]])
        primaton = Primaton(positions, 1.0)
        # First two nodes should be connected, third should not
        assert primaton.graph.has_edge(0, 1)
        assert not primaton.graph.has_edge(0, 2)
    
    def test_primatron_geometric_data(self):
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        primaton = Primaton(positions, 1.5)
        assert (0, 1) in primaton.direction_vectors
        assert (1, 0) in primaton.direction_vectors
        assert (0, 1) in primaton.rotation_operators