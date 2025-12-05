# MIGRATION PATH — V1.2 → V2 → Field Pur
Vue d'Ensemble
V1.2 (2025-2027)
    ↓
V2.0 (2027-2029)
    ↓
V3.0 Field Pur (2029+)
V1.2 → V2.0
Changements Prévus
1. Graduation de stubs:
   - aleatoric: 0 → modèle appris
   - model: 0 → ensemble de modèles
   - adversarial: heuristique → détecteur ML

2. Horizons:
   - H=2 fixe → H dynamique par pattern

3. Observation Policy:
   - Random → Policy apprise (elle-même un pattern)

4. Field Memory:
   - Storage → Vraie intégrale avec influence bidirectionnelle
Migration
def migrate_v12_to_v20(pattern_state: V12State) -> V20State:
    """
    Migration sans perte de données.
    """
    return V20State(
        # Données préservées
        nh_id=pattern_state.nh_id,
        lineage=pattern_state.lineage,
        contribution=pattern_state.contribution,
        
        # Upgraded
        uncertainty=upgrade_uncertainty(pattern_state.uncertainty),
        horizons=expand_horizons(pattern_state.horizons),
        
        # Nouveaux champs avec defaults
        observation_priority=1.0,
        field_influence=compute_initial_influence(pattern_state),
    )
V2.0 → V3.0 (Field Pur)
Changements Fondamentaux
1. Identité:
   - nh_id comme label → Pattern comme région de cohérence
   
2. Temps:
   - tau_rate comme paramètre → Temps propre émergent
   
3. Critics:
   - Composants → Régions du champ qui influencent la dynamique
   
4. Protocole:
   - Spec externe → Pattern le plus stable du champ
Migration
V2 patterns deviennent des "legacy patterns" dans V3.
Ils continuent à fonctionner via une couche de compatibilité.
Progressivement, ils sont absorbés dans la dynamique pure du champ.
Garanties de Continuité
Ce Qui Ne Change Jamais
1. Sémantique de base:
   - Report = observation d'état
   - Gradient = direction d'évolution
   - Contribution = skin in the game

2. Propriétés:
   - Récursivité (tout est pattern)
   - Multi-horizon
   - Incertitude structurée

3. Moat:
   - Data accumulée reste valable
   - Réputation transférable
   - Contribution score préservé
Ce Qui Peut Changer
1. Implémentation:
   - Comment les Critics calculent
   - Comment Field Memory stocke
   - Comment tau_rate est dérivé

2. Paramètres:
   - Nombre d'horizons
   - Seuils de lifecycle
   - Poids dans les formules

3. Modules:
   - Nouveaux modules ajoutés
   - Modules obsolètes retirés
   - Modules fusionnés