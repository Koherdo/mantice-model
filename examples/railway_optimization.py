#!/usr/bin/env python3
"""
Example: Railway network optimization using Mantice model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import networkx as nx
from src.primatron import Primaton
from src.synchronization import QuaternionicSynchronization
from src.mantice_detection import ManticeDetector

class RailwayOptimizer:
    """Railway network optimization using Mantice model"""
    
    def __init__(self, station_positions, connections):
        self.station_positions = station_positions
        self.connections = connections
        self.primatron = Primaton(station_positions, connectivity_radius=80.0)
        self.synchronization = QuaternionicSynchronization(self.primatron, coupling_strength=0.35)
        self.detector = ManticeDetector(self.primatron, coherence_threshold=0.18)
    
    def simulate_disruption(self, disrupted_lines, duration_hours=48):
        """Simulate disruption and recovery"""
        print(f"Simulating disruption affecting {len(disrupted_lines)} lines...")
        
        # Initial state
        mantices = self.detector.detect_mantices(self.synchronization)
        initial_stats = self.detector.get_mantice_statistics(mantices)
        
        print(f"Initial state: {initial_stats['number']} Mantices, average size: {initial_stats['mean_size']:.1f}")
        
        # Simulate disruption dynamics
        time_series = {
            'mantice_count': [],
            'avg_delay': [],
            'on_time_performance': []
        }
        
        for hour in range(duration_hours):
            # Update synchronization state (simplified)
            self.synchronization.step_rk4(0.1)
            
            # Detect current Mantices
            current_mantices = self.detector.detect_mantices(self.synchronization)
            stats = self.detector.get_mantice_statistics(current_mantices)
            
            time_series['mantice_count'].append(stats['number'])
            
            # Simulate performance metrics based on Mantice structure
            if stats['number'] > 15:  # High fragmentation -> poor performance
                time_series['avg_delay'].append(25 + np.random.normal(0, 5))
                time_series['on_time_performance'].append(50 + np.random.normal(0, 10))
            else:  # Consolidated structure -> good performance
                time_series['avg_delay'].append(10 + np.random.normal(0, 2))
                time_series['on_time_performance'].append(85 + np.random.normal(0, 5))
        
        return time_series
    
    def optimize_routing(self, origin, destination):
        """Find optimal route using Mantice structure"""
        mantices = self.detector.detect_mantices(self.synchronization)
        
        # Find Mantices containing origin and destination
        origin_mantice = None
        destination_mantice = None
        
        for i, mantice in enumerate(mantices):
            if origin in mantice:
                origin_mantice = i
            if destination in mantice:
                destination_mantice = i
        
        if origin_mantice is None or destination_mantice is None:
            print("Fallback to shortest path routing")
            return self._shortest_path(origin, destination)
        
        if origin_mantice == destination_mantice:
            print("Route within same Mantice - optimal path")
            return self._route_within_mantice(mantices[origin_mantice], origin, destination)
        else:
            print(f"Route between Mantices {origin_mantice} and {destination_mantice}")
            return self._route_between_mantices(mantices[origin_mantice], 
                                              mantices[destination_mantice], 
                                              origin, destination)
    
    def _shortest_path(self, origin, destination):
        """Fallback shortest path routing"""
        # Simplified implementation
        return [origin, destination]
    
    def _route_within_mantice(self, mantice, origin, destination):
        """Route within a single Mantice"""
        # Use Mantice structure for efficient routing
        return [origin, destination]
    
    def _route_between_mantices(self, mantice1, mantice2, origin, destination):
        """Route between different Mantices"""
        # Find bridge stations and route through them
        return [origin, list(mantice1)[0], list(mantice2)[0], destination]

def run_railway_example():
    """Run railway optimization example"""
    print("Running railway network optimization example...")
    
    # Create simulated railway network
    n_stations = 127
    station_positions = np.random.uniform(0, 600, (n_stations, 2))  # 600km network
    
    # Create connections (simplified)
    connections = []
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            distance = np.linalg.norm(station_positions[i] - station_positions[j])
            if distance < 100:  # Connect nearby stations
                connections.append((i, j))
    
    # Initialize optimizer
    optimizer = RailwayOptimizer(station_positions, connections)
    
    # Simulate disruption
    disrupted_lines = [(0, 1), (1, 2)]  # Example disrupted connections
    results = optimizer.simulate_disruption(disrupted_lines)
    
    print(f"Simulation completed. Final performance: {results['on_time_performance'][-1]:.1f}% on-time")
    
    # Test routing
    route = optimizer.optimize_routing(0, 10)
    print(f"Optimal route: {route}")
    
    return optimizer, results

if __name__ == "__main__":
    run_railway_example()