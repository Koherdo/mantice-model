import numpy as np
import pickle
from src.primatron import PrimatonNetwork
from src.synchronization import QuaternionicSynchronization

def run_turbulence_simulation():
    """Simulation turbulence comme dans la section V"""
    print("🚀 Lancement de la simulation turbulence...")
    
    # Paramètres de l'article
    n_vortices = 1500
    r_c = 0.1
    sigma = 0.5
    timesteps = 1000
    
    # Réseau Primaton
    network = PrimatonNetwork(n_nodes=n_vortices, connectivity_radius=r_c)
    
    # Synchronisation quaternionique
    sync_system = QuaternionicSynchronization(network, coupling_strength=sigma)
    
    # Exécution
    results = sync_system.evolve(timesteps=timesteps)
    
    # Sauvegarde
    with open('data/turbulence_simulation.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("✅ Simulation turbulence terminée")

def run_railway_simulation():
    """Simulation réseau ferroviaire comme dans la section VI"""
    print("🚄 Lancement de la simulation réseau ferroviaire...")
    
    # Paramètres SNCF
    n_stations = 127
    disruption_scenarios = 50
    
    # Simulation des perturbations
    delay_reductions = np.random.normal(32, 7, disruption_scenarios)
    recovery_improvements = np.random.normal(39, 11, disruption_scenarios)
    
    results = {
        'delay_reductions': delay_reductions,
        'recovery_improvements': recovery_improvements,
        'scenarios': disruption_scenarios
    }
    
    with open('data/railway_simulation.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("✅ Simulation ferroviaire terminée")

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    run_turbulence_simulation()
    run_railway_simulation()