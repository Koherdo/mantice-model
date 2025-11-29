import pytest
import numpy as np
import sys
import os

# Ajouter le chemin source au Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.quaternions import Quaternion, quaternion_exp, quaternion_sin


class TestQuaternions:
    def test_quaternion_creation(self):
        q = Quaternion(1, 2, 3, 4)
        assert q.w == 1
        assert q.x == 2
        assert q.y == 3
        assert q.z == 4
    
    def test_quaternion_norm(self):
        q = Quaternion(1, 1, 1, 1)
        assert abs(q.norm() - 2.0) < 1e-10
    
    def test_quaternion_multiplication(self):
        q1 = Quaternion(1, 0, 0, 0)  # Identity
        q2 = Quaternion(0, 1, 0, 0)  # i
        q3 = q1 * q2
        assert q3.w == 0
        assert q3.x == 1
        assert q3.y == 0
        assert q3.z == 0
    
    def test_quaternion_conjugate(self):
        q = Quaternion(1, 2, 3, 4)
        q_conj = q.conjugate()
        assert q_conj.w == 1
        assert q_conj.x == -2
        assert q_conj.y == -3
        assert q_conj.z == -4
    
    def test_quaternion_inverse(self):
        q = Quaternion(1, 0, 0, 0)
        q_inv = q.inverse()
        assert abs(q_inv.w - 1) < 1e-10
    
    def test_quaternion_exp(self):
        q = Quaternion(0, 0, 0, 0)  # Zero quaternion
        q_exp = quaternion_exp(q)
        assert abs(q_exp.w - 1) < 1e-10