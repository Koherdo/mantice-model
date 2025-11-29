import numpy as np
from typing import List, Callable
from .quaternions import Quaternion, quaternion_sin
from .primatron import Primaton


class QuaternionicSynchronization:
    """Non-Abelian synchronization dynamics."""

    def __init__(self, primaton: Primaton, coupling_strength: float = 0.5):
        self.primatron = primaton
        self.sigma = coupling_strength
        self.n_nodes = primaton.n_nodes

        # Initialize calendars with random states
        self.calendars = self._initialize_calendars()

        # Intrinsic frequencies (quaternionic)
        self.omega = self._initialize_frequencies()

        # Noise parameters
        self.noise_strength = 0.1
        self.noise_correlation = 0.0

    def _initialize_calendars(self) -> List[Quaternion]:
        """Initialize quaternionic calendars with random states."""
        calendars = []
        for _ in range(self.n_nodes):
            # Random unit quaternion
            random_vec = np.random.normal(0, 1, 4)
            random_vec = random_vec / np.linalg.norm(random_vec)
            calendars.append(Quaternion(*random_vec))
        return calendars

    def _initialize_frequencies(self) -> List[Quaternion]:
        """Initialize intrinsic quaternionic frequencies."""
        frequencies = []
        for _ in range(self.n_nodes):
            # Small random frequencies
            freq_vec = np.random.normal(0, 0.1, 4)
            frequencies.append(Quaternion(*freq_vec))
        return frequencies

    def coupling_function(self, i: int, j: int) -> Quaternion:
        """Non-Abelian coupling function F(C_j, C_i, r_ij)."""
        C_i = self.calendars[i]
        C_j = self.calendars[j]

        # Get rotation operator for this edge
        R_ij = self.primatron.rotation_operators[(i, j)]
        R_inv = R_ij.inverse()

        # Quaternionic sine of difference
        delta_C = C_j - C_i
        sin_delta = quaternion_sin(delta_C)

        # Apply rotation: R·sin(C_j - C_i)·R^{-1}
        coupled = R_ij * sin_delta * R_inv

        return coupled

    def dynamics_rhs(self, t: float, calendars_array: np.ndarray) -> np.ndarray:
        """Right-hand side of synchronization dynamics."""
        # Convert array back to quaternions
        self._array_to_calendars(calendars_array)

        derivatives = []

        for i in range(self.n_nodes):
            # Intrinsic dynamics
            derivative = self.omega[i]

            # Coupling term
            neighbors = self.primatron.get_neighbors(i)
            k_i = len(neighbors)

            if k_i > 0:
                coupling_sum = Quaternion(0, 0, 0, 0)
                for j in neighbors:
                    coupling_sum = coupling_sum + self.coupling_function(i, j)

                coupling_term = Quaternion(
                    self.sigma * coupling_sum.w / k_i,
                    self.sigma * coupling_sum.x / k_i,
                    self.sigma * coupling_sum.y / k_i,
                    self.sigma * coupling_sum.z / k_i,
                )
                derivative = derivative + coupling_term

            # Add noise
            noise = np.random.normal(0, self.noise_strength, 4)
            derivative = derivative + Quaternion(*noise)

            derivatives.extend([derivative.w, derivative.x, derivative.y, derivative.z])

        return np.array(derivatives)

    def _calendars_to_array(self) -> np.ndarray:
        """Convert calendars to flat array for ODE solver."""
        array = []
        for cal in self.calendars:
            array.extend([cal.w, cal.x, cal.y, cal.z])
        return np.array(array)

    def _array_to_calendars(self, array: np.ndarray):
        """Convert flat array back to quaternions."""
        for i in range(self.n_nodes):
            idx = i * 4
            self.calendars[i] = Quaternion(*array[idx : idx + 4])

    def step_rk4(self, dt: float):
        """Perform one RK4 time step."""
        current_state = self._calendars_to_array()

        k1 = self.dynamics_rhs(0, current_state)
        k2 = self.dynamics_rhs(dt / 2, current_state + dt * k1 / 2)
        k3 = self.dynamics_rhs(dt / 2, current_state + dt * k2 / 2)
        k4 = self.dynamics_rhs(dt, current_state + dt * k3)

        new_state = current_state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self._array_to_calendars(new_state)

    def get_coherence_matrix(self) -> np.ndarray:
        """Compute pairwise coherence matrix."""
        coherence = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    diff = self.calendars[i] - self.calendars[j]
                    coherence[i, j] = diff.norm()
        return coherence

    def get_global_order_parameter(self) -> float:
        """Compute global order parameter R."""
        # Extract scalar parts for order parameter calculation
        scalar_parts = [cal.w for cal in self.calendars]
        complex_phases = [np.exp(1j * phase) for phase in scalar_parts]
        return np.abs(np.mean(complex_phases))