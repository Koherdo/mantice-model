import numpy as np
from typing import Tuple


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Quaternion(
            self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
        )

    def __sub__(self, other):
        return Quaternion(
            self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z
        )

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def to_array(self):
        return np.array([self.w, self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2], arr[3])


def quaternion_exp(q: Quaternion) -> Quaternion:
    """Exponentielle d'un quaternion"""
    v_norm = np.sqrt(q.x**2 + q.y**2 + q.z**2)
    if v_norm < 1e-10:
        return Quaternion(np.exp(q.w), 0, 0, 0)

    exp_w = np.exp(q.w)
    return Quaternion(
        exp_w * np.cos(v_norm),
        exp_w * (q.x / v_norm) * np.sin(v_norm),
        exp_w * (q.y / v_norm) * np.sin(v_norm),
        exp_w * (q.z / v_norm) * np.sin(v_norm),
    )


def rotation_operator(angle: float, axis: np.ndarray) -> Quaternion:
    """Opérateur de rotation quaternionique"""
    axis_norm = axis / np.linalg.norm(axis)
    return Quaternion(
        np.cos(angle / 2),
        axis_norm[0] * np.sin(angle / 2),
        axis_norm[1] * np.sin(angle / 2),
        axis_norm[2] * np.sin(angle / 2),
    )
