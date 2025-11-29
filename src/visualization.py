import matplotlib.pyplot as plt
import numpy as np
from typing import List, Set, Dict, Any
import sys
import os

# Ajouter le chemin pour résoudre les imports relatifs
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available, 3D visualization disabled")


class ManticeVisualizer:
    """Visualization tools for Mantice model."""

    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_phase_transition(
        self,
        sigma_values: np.ndarray,
        order_parameters: list,
        critical_sigma: float,
        save_path: str = None,
    ):
        """Plot phase transition with finite-size scaling"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Main phase transition plot
        system_sizes = [100, 500, 1000, 5000, 10000]
        for i, R_values in enumerate(order_parameters):
            ax1.plot(
                sigma_values,
                R_values,
                "o-",
                label=f"N={system_sizes[i]}",
                color=self.colors[i],
                markersize=4,
            )

        ax1.axvline(
            critical_sigma,
            color="red",
            linestyle="--",
            label=f"$\\sigma_c = {critical_sigma:.3f}$",
        )
        ax1.set_xlabel("Coupling Strength $\\sigma$")
        ax1.set_ylabel("Order Parameter $R$")
        ax1.set_title("Phase Transition")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Finite-size scaling inset
        ax2.text(
            0.5,
            0.5,
            "Finite-Size Scaling\n(Collapse plot)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_xlabel("$(\\sigma - \\sigma_c)N^{1/\\nu}$")
        ax2.set_ylabel("$R N^{\\beta/\\nu}$")
        ax2.set_title("Finite-Size Scaling Collapse")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_phase_transition_comprehensive(
        self, results: Dict, critical_sigma: float, save_path: str = None
    ):
        """Plot comprehensive phase transition with multiple system sizes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Main phase transition plot
        for i, (N, data) in enumerate(results.items()):
            sigma = data["sigma"]
            R = data["R"]
            susceptibility = data["susceptibility"]

            ax1.plot(sigma, R, "o-", label=f"N={N}", color=self.colors[i], markersize=4)
            ax2.plot(
                sigma,
                susceptibility,
                "s-",
                label=f"N={N}",
                color=self.colors[i],
                markersize=4,
            )

        ax1.axvline(
            critical_sigma,
            color="red",
            linestyle="--",
            label=f"$\\sigma_c = {critical_sigma:.3f}$",
        )
        ax1.set_xlabel("Disorder Strength ($\\sigma$)")
        ax1.set_ylabel("Order Parameter ($R$)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Disorder Strength ($\\sigma$)")
        ax2.set_ylabel("Susceptibility ($\\chi$)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_transport_scaling(
        self,
        distances: list,
        mantice_times: list,
        diffusive_times: list,
        save_path: str = None,
    ):
        """Plot transport scaling comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Linear scale
        ax1.loglog(
            distances,
            mantice_times,
            "bo-",
            label="Mantice $T \\sim \\log d$",
            linewidth=2,
        )
        ax1.loglog(
            distances,
            diffusive_times,
            "ro-",
            label="Diffusion $T \\sim d^2$",
            linewidth=2,
        )
        ax1.set_xlabel("Distance $d$")
        ax1.set_ylabel("Transport Time $T$")
        ax1.set_title("Transport Scaling Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Log-linear scale to show logarithmic scaling
        ax2.semilogy(
            np.log(distances), mantice_times, "bo-", label="Mantice", linewidth=2
        )
        ax2.semilogy(
            np.log(distances), diffusive_times, "ro-", label="Diffusion", linewidth=2
        )
        ax2.set_xlabel("$\\log d$")
        ax2.set_ylabel("Transport Time $T$")
        ax2.set_title("Logarithmic Scaling Verification")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_energy_spectrum(
        self,
        k_values: list,
        E_mantice: list,
        E_les: list,
        E_dns: list,
        save_path: str = None,
    ):
        """Plot turbulence energy spectrum"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Energy spectrum
        ax1.loglog(k_values, E_dns, "k-", label="DNS", linewidth=2)
        ax1.loglog(k_values, E_mantice, "b-", label="Mantice", linewidth=2)
        ax1.loglog(k_values, E_les, "r-", label="LES", linewidth=2)

        # Kolmogorov -5/3 line
        k_range = np.array([k_values[10], k_values[-10]])
        E_kolmogorov = k_range ** (-5 / 3) * E_dns[10] / k_values[10] ** (-5 / 3)
        ax1.loglog(k_range, E_kolmogorov, "k--", label="-5/3 slope", alpha=0.7)

        ax1.set_xlabel("Wavenumber $k$")
        ax1.set_ylabel("Energy Spectrum $E(k)$")
        ax1.set_title("Turbulence Energy Spectrum")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Compensated spectrum
        ax2.loglog(
            k_values,
            E_dns * np.array(k_values) ** (5 / 3),
            "k-",
            label="DNS",
            linewidth=2,
        )
        ax2.loglog(
            k_values,
            E_mantice * np.array(k_values) ** (5 / 3),
            "b-",
            label="Mantice",
            linewidth=2,
        )
        ax2.loglog(
            k_values,
            E_les * np.array(k_values) ** (5 / 3),
            "r-",
            label="LES",
            linewidth=2,
        )

        ax2.set_xlabel("Wavenumber $k$")
        ax2.set_ylabel("$E(k) k^{5/3}$")
        ax2.set_title("Compensated Spectrum")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_railway_recovery(self, time_series: dict, save_path: str = None):
        """Plot railway network recovery during disruption"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        time = time_series["time"]

        # Delayed trains
        ax1.plot(
            time,
            time_series["delayed_trains_mantice"],
            "b-",
            label="Mantice",
            linewidth=2,
        )
        ax1.plot(
            time,
            time_series["delayed_trains_network_flow"],
            "r-",
            label="Network Flow",
            linewidth=2,
        )
        ax1.plot(
            time,
            time_series["delayed_trains_greedy"],
            "g-",
            label="Greedy",
            linewidth=2,
        )
        ax1.set_ylabel("Number of Delayed Trains")
        ax1.set_title("Railway Network Recovery - Storm Ciaran")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Average delay
        ax2.plot(
            time, time_series["avg_delay_mantice"], "b-", label="Mantice", linewidth=2
        )
        ax2.plot(
            time,
            time_series["avg_delay_network_flow"],
            "r-",
            label="Network Flow",
            linewidth=2,
        )
        ax2.plot(
            time, time_series["avg_delay_greedy"], "g-", label="Greedy", linewidth=2
        )
        ax2.set_ylabel("Average Delay (min)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Mantice count
        ax3.plot(time, time_series["mantice_count"], "purple", linewidth=2)
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Number of Mantices")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
