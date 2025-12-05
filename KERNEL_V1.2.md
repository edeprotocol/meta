# KERNEL V1.2 — Spécification Invariante

## Statut

Ce document définit le **noyau invariant** du Synthetic Field Layer.
Ces structures ne changent pas entre versions mineures.
Toute extension doit être compatible avec ce kernel.

## Structures de Données

### Report

Un Report est un échantillon de l'état d'un pattern à un instant donné.
Report:
nh_id       : bytes32              # Handle sur le pattern (dérivé recommandé)
state       : Tensor[d_s]          # État observé
action      : Tensor[d_a]          # Action exécutée
cost        : Tensor[d_c]          # Coût consommé (compute, énergie, etc.)
outcome     : Tensor[d_o]          # Résultat/feedback
timestamp   : uint64               # Horodatage physique
lineage     : bytes32[]            # Ancêtres directs (fork/merge history)


**Recommandation pour nh_id :**
nh_id = hash(lineage_root, creation_timestamp, initial_state_hash)


### GradientPacket

Un GradientPacket est la réponse du champ à un pattern.
GradientPacket:
nh_id        : bytes32             # Pattern concerné
param_grad   : Tensor[d_p, H]      # Gradient multi-horizon
horizons     : float[H]            # Valeurs des horizons (ex: [1.0, 10.0])
alloc_signal : AllocSignal         # Signal d'allocation
uncertainty  : UncertaintyBundle   # Incertitude structurée
critic_ids   : bytes32[]           # Critics ayant contribué
timestamp    : uint64              # Horodatage du gradient


### AllocSignal

Signal d'allocation de ressources (temps propre).
AllocSignal:
tau_rate               : float     # Temps propre / temps physique
allowed_envs           : bytes32[] # Environnements autorisés
compliance_window      : uint64    # Délai pour appliquer le signal
non_compliance_penalty : float     # Pénalité si ignoré
Sémantique de tau_rate:
tau_rate = 0     : pattern gelé (pas de compute)
tau_rate ∈ (0,1) : pattern ralenti
tau_rate = 1     : vitesse nominale
tau_rate > 1     : pattern accéléré


### UncertaintyBundle

Incertitude structurée sur le gradient.
UncertaintyBundle:
epistemic   : float    # Désaccord entre critics
aleatoric   : float    # Bruit estimé des données
model       : float    # Incertitude structurelle du modèle
adversarial : float    # Suspicion de manipulation
scalar      : float    # Agrégation pour consommateurs simples


### ContributionScore

Contribution d'un pattern à la mémoire collective.
ContributionScore:
reports_submitted : uint64   # Nombre de Reports soumis
gradient_utility  : float    # Utilité des gradients dérivés
data_uniqueness   : float    # Entropie relative des contributions
access_level      : uint8    # Niveau d'accès (0-3)


### LifecycleEvent

Événement dans le cycle de vie d'un pattern.
LifecycleEvent:
nh_id      : bytes32
event_type : enum { EMERGE, FORK, MERGE, FREEZE, DISSOLVE }
timestamp  : uint64
parent_ids : bytes32[]    # Pour FORK/MERGE
metadata   : bytes        # Données additionnelles


## Propriétés Invariantes

1. **Tenseurs uniquement** — Pas de JSON, pas de texte dans le kernel
2. **Gradients vectoriels** — Pas de scores scalaires
3. **Multi-critics** — Minimum 2 critics, `critic_ids` toujours renseigné
4. **Récursivité** — Critics et encodeurs sont eux-mêmes des patterns
5. **Lineage** — Tout pattern a une histoire (même vide)

## Interface Minimale

```python
class SyntheticFieldKernel:
    def report(self, report: Report) -> None
    def pull_gradient(self, nh_id: bytes32) -> GradientPacket
    def get_contribution(self, nh_id: bytes32) -> ContributionScore
    def emit_lifecycle_event(self, event: LifecycleEvent) -> None
```