# Synthetic Field Layer — V0 Implementation
Statut
Ceci est un émulateur discret du Synthetic Field Layer.
Il implémente le Kernel V1.2 et le Bundle V1.2 de manière simplifiée, suffisante pour :

Valider les concepts
Brancher des systèmes IA réels
Collecter des données pour améliorer les Critics
Ce que V0 fait
✅ Reports et GradientPackets (tenseurs via numpy/torch)
✅ 2 horizons (short/long)
✅ Multi-critics (2 critics minimum)
✅ Uncertainty.epistemic (désaccord critics)
✅ tau_rate allocation
✅ ContributionScore basique
✅ Field Memory (stockage Parquet)
✅ Lineage tracking basique
Ce que V0 ne fait pas (stubs)
⚠️ Uncertainty.aleatoric = 0
⚠️ Uncertainty.model = 0
⚠️ Uncertainty.adversarial = heuristique simple
⚠️ Observation Policy = uniform random
⚠️ Protocol Evolution = version fixe
Démarrage
pip install -r requirements.txt
python field_kernel.py --config config.yaml
Usage
from sfl import FieldClient

# Connexion
client = FieldClient("localhost:50051")

# Enregistrement
handle = client.register(param_shape=(1024,))

# Boucle principale
while True:
    state, action, cost, outcome = my_system.step()
    
    # Report
    client.report(handle, state, action, cost, outcome)
    
    # Pull gradient
    grad = client.pull_gradient(handle)
    
    # Appliquer
    my_system.apply_gradient(grad.param_grad[:, 0])  # short horizon
    
    # Ajuster vitesse selon tau_rate
    my_system.set_speed(grad.alloc_signal.tau_rate)