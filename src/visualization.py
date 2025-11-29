import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from typing import List, Set
from .primatron import Primaton

class ManticeVisualizer:
    """Visualization tools for Mantice model"""
    
    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_phase_transition(self, sigma_values: list, order_parameters: list, 
                            critical_sigma: float, save_path: str = None):
        """Plot phase transition with finite-size scaling"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Main phase transition plot
        for i, (sigma, R) in enumerate(zip(sigma_values, order_parameters)):
            ax1.plot(sigma, R, 'o-', label=f'N={100*(i+1)}', color=self.colors[i])
        
        ax1.axvline(critical_sigma, color='red', linestyle='--', 
                   label=f'$\\sigma_c = {critical_sigma:.3f}$')
        ax1.set_xlabel('Coupling Strength $\\sigma$')
        ax1.set_ylabel('Order Parameter $R$')
        ax1.set_title('Phase Transition')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Finite-size scaling inset
        ax2.set_xlabel('$(\\sigma - \\sigma_c)N^{1/\\nu}$')
        ax2.set_ylabel('$R N^{\\beta/\\nu}$')
        ax2.set_title('Finite-Size Scaling Collapse')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_transport_scaling(self, distances: list, mantice_times: list, 
                             diffusive_times: list, save_path: str = None):
        """Plot transport scaling comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear scale
        ax1.loglog(distances, mantice_times, 'bo-', label='Mantice $T \\sim \\log d$', linewidth=2)
        ax1.loglog(distances, diffusive_times, 'ro-', label='Diffusion $T \\sim d^2$', linewidth=2)
        ax1.set_xlabel('Distance $d$')
        ax1.set_ylabel('Transport Time $T$')
        ax1.set_title('Transport Scaling Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log-linear scale to show logarithmic scaling
        ax2.semilogy(np.log(distances), mantice_times, 'bo-', label='Mantice', linewidth=2)
        ax2.semilogy(np.log(distances), diffusive_times, 'ro-', label='Diffusion', linewidth=2)
        ax2.set_xlabel('$\\log d$')
        ax2.set_ylabel('Transport Time $T$')
        ax2.set_title('Logarithmic Scaling Verification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_energy_spectrum(self, k_values: list, E_mantice: list, E_les: list, 
                           E_dns: list, save_path: str = None):
        """Plot turbulence energy spectrum"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Energy spectrum
        ax1.loglog(k_values, E_dns, 'k-', label='DNS', linewidth=2)
        ax1.loglog(k_values, E_mantice, 'b-', label='Mantice', linewidth=2)
        ax1.loglog(k_values, E_les, 'r-', label='LES', linewidth=2)
        
        # Kolmogorov -5/3 line
        k_range = np.array([k_values[10], k_values[-10]])
        E_kolmogorov = k_range**(-5/3) * E_dns[10] / k_values[10]**(-5/3)
        ax1.loglog(k_range, E_kolmogorov, 'k--', label='-5/3 slope', alpha=0.7)
        
        ax1.set_xlabel('Wavenumber $k$')
        ax1.set_ylabel('Energy Spectrum $E(k)$')
        ax1.set_title('Turbulence Energy Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Compensated spectrum
        ax2.loglog(k_values, E_dns * k_values**(5/3), 'k-', label='DNS', linewidth=2)
        ax2.loglog(k_values, E_mantice * k_values**(5/3), 'b-', label='Mantice', linewidth=2)
        ax2.loglog(k_values, E_les * k_values**(5/3), 'r-', label='LES', linewidth=2)
        
        ax2.set_xlabel('Wavenumber $k$')
        ax2.set_ylabel('$E(k) k^{5/3}$')
        ax2.set_title('Compensated Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_railway_recovery(self, time_series: dict, save_path: str = None):
        """Plot railway network recovery during disruption"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        time = time_series['time']
        
        # Delayed trains
        ax1.plot(time, time_series['delayed_trains_mantice'], 'b-', label='Mantice', linewidth=2)
        ax1.plot(time, time_series['delayed_trains_network_flow'], 'r-', label='Network Flow', linewidth=2)
        ax1.plot(time, time_series['delayed_trains_greedy'], 'g-', label='Greedy', linewidth=2)
        ax1.set_ylabel('Number of Delayed Trains')
        ax1.set_title('Railway Network Recovery - Storm Ciaran')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average delay
        ax2.plot(time, time_series['avg_delay_mantice'], 'b-', label='Mantice', linewidth=2)
        ax2.plot(time, time_series['avg_delay_network_flow'], 'r-', label='Network Flow', linewidth=2)
        ax2.plot(time, time_series['avg_delay_greedy'], 'g-', label='Greedy', linewidth=2)
        ax2.set_ylabel('Average Delay (min)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mantice count
        ax3.plot(time, time_series['mantice_count'], 'purple', linewidth=2)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Number of Mantices')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_3d_mantices(self, primaton: Primaton, mantices: List[Set[int]], 
                        save_path: str = None):
        """Create 3D visualization of Mantices"""
        fig = go.Figure()
        
        positions = primaton.positions
        
        # Plot each Mantice with different color
        for i, mantice in enumerate(mantices):
            if len(mantice) < 2:
                continue
                
            mantice_nodes = list(mantice)
            x = [positions[node][0] for node in mantice_nodes]
            y = [positions[node][1] for node in mantice_nodes]
            z = [positions[node][2] for node in mantice_nodes]
            
            color = self.colors[i % len(self.colors)]
            color_str = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=8, color=color_str),
                name=f'Mantice {i+1} (size: {len(mantice)})'
            ))
            
            # Add edges within Mantice
            for node_i in mantice_nodes:
                for node_j in mantice_nodes:
                    if node_i < node_j and primaton.graph.has_edge(node_i, node_j):
                        edge_x = [positions[node_i][0], positions[node_j][0]]
                        edge_y = [positions[node_i][1], positions[node_j][1]]
                        edge_z = [positions[node_i][2], positions[node_j][2]]
                        
                        fig.add_trace(go.Scatter3d(
                            x=edge_x, y=edge_y, z=edge_z,
                            mode='lines',
                            line=dict(color=color_str, width=2),
                            showlegend=False
                        ))
        
        fig.update_layout(
            title='3D Mantice Structure Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()