import matplotlib.pyplot as plt
import numpy as np
from typing import List, Set
import sys
import os

# Ajouter le chemin pour résoudre les imports relatifs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available, 3D visualization disabled")

from primatron import Primaton
from quaternions import Quaternion


class ManticeVisualizer:
    """Visualization tools for Mantice model."""
    
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
    
    # ... (le reste du code reste identique mais avec les imports corrigés)

    def plot_3d_mantices(self, primaton: Primaton, mantices: List[Set[int]], 
                        save_path: str = None):
        """Create 3D visualization of Mantices"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for 3D visualization")
            return
            
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