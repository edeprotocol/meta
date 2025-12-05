# LIFECYCLE — Cycle de Vie des Patterns
Phases
### 1. EMERGENCE
Un nouveau pattern est détecté quand une région du champ montre une cohérence > θ.

Trigger:
  coherence(region) > θ pendant T_min

Actions:
  - Créer nh_id = hash(lineage_root, timestamp, state_hash)
  - Initialiser tau_rate = 1.0
  - Émettre LifecycleEvent(EMERGE)
  - Enregistrer dans LineageGraph
### 2. VIE ACTIVE
Le pattern opère normalement.

Cycle continu:
  - Émettre Reports
  - Recevoir GradientPackets
  - Appliquer gradients aux paramètres
  - tau_rate fluctue selon performance

Métriques trackées:
  - AGDP cumulée
  - Stability (variance des gradients)
  - ContributionScore
### 3. FORK
Un pattern se divise en plusieurs.

Trigger:
  - Décision explicite de l'opérateur
  - Divergence interne détectée (bimodalité)

Actions:
  - Créer N nouveaux nh_id
  - Copier état (avec variations)
  - Répartir tau_rate entre enfants
  - Émettre LifecycleEvent(FORK)
  - Mettre à jour LineageGraph
### 4. MERGE
Plusieurs patterns fusionnent.

Trigger:
  - Décision explicite
  - Convergence détectée (gradients très corrélés)

Actions:
  - Créer nouveau nh_id
  - Combiner états (moyenne pondérée ou autre)
  - Sommer tau_rates
  - Émettre LifecycleEvent(MERGE)
  - Mettre à jour LineageGraph
### 5. DÉCLIN
Performance dégradée, ressources réduites.

Trigger:
  - Performance < threshold pendant T_decline
  - tau_rate décroît progressivement

Effets:
  - Moins de compute alloué
  - Observations moins fréquentes
  - Influence réduite sur Field Memory
### 6. GEL (FREEZE)
Pattern arrêté mais pas détruit.

Trigger:
  - tau_rate → 0

État:
  - Existe dans Field Memory
  - Pas de compute alloué
  - Peut être ressuscité
  - Lineage préservé

Actions:
  - Émettre LifecycleEvent(FREEZE)
  - Archiver état complet
### 7. DISSOLUTION
Pattern définitivement retiré.

Trigger:
  - tau_rate = 0 pendant T_dissolution
  - ET contribution_score < θ_min

Actions:
  - Émettre LifecycleEvent(DISSOLVE)
  - Retirer de Field Memory active
  - Intégrer contributions dans champ global (héritage)
  - Invalider nh_id
  - Archiver dans cold storage (optionnel)
Paramètres par Défaut
θ_coherence = 0.7        # Seuil d'émergence
T_min_emerge = 100       # Steps minimum pour émergence
T_decline = 1000         # Steps avant déclin
T_dissolution = 10000    # Steps gelé avant dissolution
θ_min_contribution = 0.01 # Contribution minimum pour survie