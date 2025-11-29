#!/usr/bin/env python3
"""
Example: Turbulence simulation using Mantice model
"""

import sys
import os

from src import primatron

# Ajouter le chemin source
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from src.primatron import Primaton
from src.synchronization import QuaternionicSynchronization
from src.mantice_detection import ManticeDetector
from src.visualization import ManticeVisualizer


def run_turbulence_simulation():
    """Run a complete turbulence simulation"""
    print("Running turbulence simulation...")
    
    # Create vortex positions (simulated turbulence data)
    n_vortices = 1500
    positions = np.random.normal(0, 1, (n_vortices, 3))
    
    # Create Primaton network
    primaton = Primaton(positions, connectivity_radius=0.1)
    print(f"Created Primaton with {primatron.n_nodes} nodes, average degree: {primatron.get_average_degree():.2f}")
    
    # Initialize synchronization dynamics
    synchronization = QuaternionicSynchronization(primatron, coupling_strength=0.5)
    
    # Run simulation
    n_steps = 1000
    dt = 0.01
    
    order_parameters = []
    mantice_statistics = []
    
    detector = ManticeDetector(primatron, coherence_threshold=0.15)
    
    for step in range(n_steps):
        synchronization.step_rk4(dt)
        
        if step % 100 == 0:
            # Compute order parameter
            R = synchronization.get_global_order_parameter()
            order_parameters.append(R)
            
            # Detect Mantices
            mantices = detector.detect_mantices(synchronization)
            stats = detector.get_mantice_statistics(mantices)
            mantice_statistics.append(stats)
            
            print(f"Step {step}: R = {R:.3f}, Mantices = {stats['number']}")
    
    # Visualize results
    visualizer = ManticeVisualizer()
    
    # Plot final Mantice structure
    final_mantices = detector.detect_mantices(synchronization)
    visualizer.plot_3d_mantices(primatron, final_mantices, 
                               save_path='../results/figures/turbulence_mantices.html')
    
    print("Simulation completed successfully!")
    return synchronization, detector


if __name__ == "__main__":
    run_turbulence_simulation()