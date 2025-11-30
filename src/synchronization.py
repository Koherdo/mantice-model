import numpy as np
import math  # AJOUT IMPORT
from typing import List, Dict, Tuple
from .primatron import (
    PrimatonNetwork,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_norm,
)


class QuaternionicSynchronization:
    """Système de synchronisation quaternionique selon les Équations 5-10"""

    def __init__(
        self,
        network: PrimatonNetwork,
        coupling_strength: float = 0.5,
        noise_intensity: float = 0.01,
        intrinsic_freq_scale: float = 1.0,
    ):
        self.network = network
        self.sigma = coupling_strength
        self.D = noise_intensity
        self.omega_scale = intrinsic_freq_scale

        # Fréquences intrinsèques aléatoires
        self.intrinsic_frequencies = self._initialize_intrinsic_frequencies()

        # Historique pour l'analyse
        self.history = {"calendars": [], "order_parameter": [], "time": []}

    def _initialize_intrinsic_frequencies(self) -> np.ndarray:
        """Initialise les fréquences intrinsèques selon une distribution gaussienne"""
        n_nodes = self.network.n_nodes
        return np.random.normal(0, self.omega_scale, (n_nodes, 4))

    def quaternion_sine(self, q: np.ndarray, n_terms: int = 3) -> np.ndarray:
        """Sinus quaternionique selon l'Équation 10 (approximation par série)"""
        # Pour les petites valeurs, on approxime sin(q) ≈ q
        if quaternion_norm(q) < 0.1:
            return q

        # Approximation avec quelques termes de la série
        result = q.copy()
        q_power = q.copy()

        for n in range(1, n_terms):
            # q^(2n+1)
            for _ in range(2):
                q_power = quaternion_multiply(q_power, q)

            # CORRECTION : Utiliser math.factorial
            term = ((-1) ** n / math.factorial(2 * n + 1)) * q_power
            result = result + term

        return result

    def non_abelian_coupling(
        self, q_i: np.ndarray, q_j: np.ndarray, geometric_data: dict
    ) -> np.ndarray:
        """Fonction de couplage non-abélienne selon l'Équation 6"""
        if not geometric_data:
            return np.zeros(4)

        # Différence des calendriers
        delta_q = q_j - q_i

        # Sinus quaternionique de la différence
        sin_delta = self.quaternion_sine(delta_q)

        # Opérateur de rotation
        R = geometric_data["rotation_operator"]
        R_inv = quaternion_conjugate(
            R
        )  # Inverse = conjugué pour les quaternions unitaires

        # Application de la rotation: R · sin(Δ) · R^{-1}
        temp = quaternion_multiply(R, sin_delta)
        result = quaternion_multiply(temp, R_inv)

        return result

    def compute_derivative(self, calendars: np.ndarray, time: float) -> np.ndarray:
        """Calcule la dérivée selon l'Équation 5"""
        n_nodes = self.network.n_nodes
        derivatives = np.zeros((n_nodes, 4))

        for i in range(n_nodes):
            # Terme de fréquence intrinsèque
            derivatives[i] += self.intrinsic_frequencies[i]

            # Terme de couplage avec les voisins
            neighbors = self.network.get_neighbors(i)
            k_i = len(neighbors) if neighbors else 1  # Éviter division par zéro

            coupling_sum = np.zeros(4)
            for j in neighbors:
                geometric_data = self.network.get_geometric_data(i, j)
                coupling_term = self.non_abelian_coupling(
                    calendars[i], calendars[j], geometric_data
                )
                coupling_sum += coupling_term

            derivatives[i] += (self.sigma / k_i) * coupling_sum

            # Bruit quaternionique (Éq. 5)
            noise = np.random.normal(0, np.sqrt(2 * self.D), 4)
            derivatives[i] += noise

        return derivatives

    def rk4_integration(
        self, calendars: np.ndarray, dt: float, time: float
    ) -> np.ndarray:
        """Intégration RK4 adaptée aux quaternions"""
        k1 = self.compute_derivative(calendars, time)
        k2 = self.compute_derivative(calendars + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = self.compute_derivative(calendars + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = self.compute_derivative(calendars + dt * k3, time + dt)

        new_calendars = calendars + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalisation pour maintenir les quaternions unitaires
        for i in range(len(new_calendars)):
            norm = quaternion_norm(new_calendars[i])
            if norm > 1e-10:
                new_calendars[i] /= norm

        return new_calendars

    def compute_order_parameter(self, calendars: np.ndarray) -> float:
        """Calcule le paramètre d'ordre global selon l'Équation 18"""
        # Extraire la partie scalaire moyenne
        scalar_parts = calendars[:, 0]  # Composante w
        complex_phases = np.exp(1j * scalar_parts)
        R = np.abs(np.mean(complex_phases))
        return R

    def evolve(self, timesteps: int = 1000, dt: float = 0.01) -> Dict:
        """Évolue le système dans le temps"""
        current_calendars = self.network.calendars.copy()
        time = 0.0

        for step in range(timesteps):
            # Intégration RK4
            current_calendars = self.rk4_integration(current_calendars, dt, time)

            # Mise à jour du réseau
            self.network.update_calendars(current_calendars)

            # Enregistrement périodique
            if step % 100 == 0:
                R = self.compute_order_parameter(current_calendars)
                self.history["calendars"].append(current_calendars.copy())
                self.history["order_parameter"].append(R)
                self.history["time"].append(time)

            time += dt

        return self.history
