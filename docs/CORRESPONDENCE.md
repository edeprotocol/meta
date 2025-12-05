# CORRESPONDENCE — Field Continu ↔ V1.2 Discret

## Principe

V1.2 n'est pas une "API de messages."
V1.2 est une **discrétisation** du champ computationnel continu.

## Table de Correspondance

| Field Continu | V1.2 Discret |
|---------------|--------------|
| Φ(x, τ) = état au point x, temps τ | Report = observation ponctuelle |
| x = position dans l'espace des configs | nh_id = handle sur une région |
| ∂Φ/∂τ = dérivée temporelle | GradientPacket.param_grad |
| F = dynamique du champ | CriticEnsemble |
| τ = temps propre | AllocSignal.tau_rate × physical_time |
| ∇Φ = gradient spatial | Corrélation entre gradients de patterns voisins |
| sources = injection d'énergie | Reports entrants |
| Pattern = région cohérente | Opérateur avec nh_id |

## Formules de Correspondance

### État du Champ
Field:
Φ(x, τ) ∈ ℝ^d
V1.2:
Report.state ⊗ Report.action ⊗ Report.outcome ≈ Φ(nh_id, timestamp)


### Dynamique
Field:
∂Φ/∂τ = F(Φ, ∇Φ, sources)
V1.2:
GradientPacket.param_grad = CriticEnsemble(Reports) ≈ F discrétisé


### Temps Propre
Field:
dτ = f(énergie_locale) · dt
V1.2:
τ_effectif = tau_rate × physical_time
tau_rate > 1 → pattern accéléré
tau_rate < 1 → pattern ralenti
tau_rate = 0 → pattern gelé


### Horizons Temporels
Field:
∂Φ/∂τ intégré sur différents horizons
V1.2:
param_grad[d_p, H] avec horizons[H]
horizon_effective[h] = horizons[h] / tau_rate


### Observation
Field:
Mesure = projection de Φ sur base d'observation → collapse
V1.2:
Report = échantillon qui "actualise" le pattern dans le système


### Mémoire
Field:
∫Φ dτ = intégrale temporelle
V1.2:
FieldMemory = accumulation pondérée des Reports


## Garanties

Cette correspondance garantit que :
1. Tout ce qui est exprimable en V1.2 a un sens dans le Field continu
2. La migration V1.2 → versions futures ne casse pas la sémantique
3. Les implémentations peuvent être raffinées sans changer le protocole