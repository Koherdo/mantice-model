import numpy as np
from typing import Tuple


class Quaternion:
    """Quaternion operations for Mantice model."""

    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.array = np.array([w, x, y, z])

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
        )

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z
        )

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(w, x, y, z)

    def norm(self) -> float:
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quaternion":
        norm_sq = self.norm() ** 2
        return Quaternion(
            self.w / norm_sq, -self.x / norm_sq, -self.y / norm_sq, -self.z / norm_sq
        )

    def normalize(self) -> "Quaternion":
        n = self.norm()
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z

        return np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ]
        )


def quaternion_exp(q: Quaternion) -> Quaternion:
    """Exponential of a quaternion."""
    v_norm = np.sqrt(q.x**2 + q.y**2 + q.z**2)
    if v_norm == 0:
        return Quaternion(np.exp(q.w), 0, 0, 0)

    exp_w = np.exp(q.w)
    return Quaternion(
        exp_w * np.cos(v_norm),
        exp_w * (q.x / v_norm) * np.sin(v_norm),
        exp_w * (q.y / v_norm) * np.sin(v_norm),
        exp_w * (q.z / v_norm) * np.sin(v_norm),
    )


def quaternion_sin(q: Quaternion, n_terms: int = 10) -> Quaternion:
    """Taylor series expansion of sine for quaternions."""
    result = Quaternion(0, 0, 0, 0)
    q_power = Quaternion(1, 0, 0, 0)

    for n in range(n_terms):
        if n % 2 == 0:  # Even terms are zero for sine
            q_power = q_power * q
            continue

        term = q_power
        if (n // 2) % 2 == 1:  # Alternating signs
            term = Quaternion(-term.w, -term.x, -term.y, -term.z)

        factorial = np.math.factorial(n)
        term = Quaternion(
            term.w / factorial,
            term.x / factorial,
            term.y / factorial,
            term.z / factorial,
        )
        result = result + term
        q_power = q_power * q

    return result


def create_rotation_quaternion(angle: float, axis: np.ndarray) -> Quaternion:
    """Create rotation quaternion from angle and axis."""
    axis = axis / np.linalg.norm(axis)
    return Quaternion(
        np.cos(angle / 2),
        axis[0] * np.sin(angle / 2),
        axis[1] * np.sin(angle / 2),
        axis[2] * np.sin(angle / 2),
    )
