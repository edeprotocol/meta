# FIELD MEMORY — L'Intégrale du Champ
Principe Fondamental
Field Memory n'est pas une base de données.
C'est l'intégrale temporelle de Φ — la mémoire vivante du champ.

M(t) = ∫₀ᵗ w(τ) · Φ(τ) dτ

où w(τ) = fonction de pondération (decay, importance, etc.)
Structure
Représentation
class FieldMemory:
    integral: Tensor           # Intégrale courante
    index: SpatialIndex        # Index pour queries rapides
    contribution_ledger: Dict  # Qui a contribué quoi
Opérations Fondamentales
def integrate(self, report: Report) -> None:
    """Ajoute une contribution à l'intégrale."""
    embedding = self.encoder(report)
    weight = self.compute_weight(report)
    self.integral += weight * embedding
    self.contribution_ledger[report.nh_id] += weight

def query(self, 
          projection: Tensor, 
          time_range: Optional[Tuple] = None,
          pattern_filter: Optional[List[bytes32]] = None) -> Tensor:
    """Projette l'intégrale sur une base."""
    relevant = self.filter(time_range, pattern_filter)
    return projection @ relevant

def influence_on_gradient(self, nh_id: bytes32) -> Tensor:
    """Calcule comment l'historique influence le gradient d'un pattern."""
    pattern_history = self.get_pattern_history(nh_id)
    global_context = self.get_global_context()
    return self.influence_model(pattern_history, global_context)
Pondération
Facteurs de Poids
def compute_weight(self, report: Report) -> float:
    w_recency = exp(-λ_time * age(report))
    w_quality = report_quality_score(report)
    w_uniqueness = information_gain(report, self.integral)
    w_contribution = self.contribution_ledger[report.nh_id].score
    
    return w_recency * w_quality * w_uniqueness * w_contribution
Decay Temporel
Observations récentes pèsent plus que anciennes :
w_recency = exp(-λ × (t_now - t_report))

λ configurable selon horizon d'intérêt
Access Control
Niveaux
Level 0 (contribution < 100 reports):
  - Query gradients agrégés seulement
  - Pas d'accès aux données brutes

Level 1 (100-1k reports):
  - Query gradients multi-horizon
  - Accès aux statistiques agrégées

Level 2 (1k-10k reports):
  - Accès aux lineage graphs
  - Query sur sous-ensembles de patterns

Level 3 (>10k reports):
  - Accès complet aux données brutes
  - Query arbitraires
  - Export autorisé
Enforcement
def query_with_access_control(self, 
                               nh_id: bytes32,
                               query: Query) -> Result:
    level = self.get_access_level(nh_id)
    if query.requires_level > level:
        raise AccessDenied(f"Requires level {query.requires_level}, have {level}")
    return self.execute_query(query)
Compaction
Problème
L'intégrale croît indéfiniment → besoin de compacter.

Solution
def compact(self, 
            keep_recent: int = 10000,
            compression_ratio: float = 0.1) -> None:
    """
    Garde les observations récentes en haute résolution.
    Compresse les anciennes en résumés.
    """
    recent = self.observations[-keep_recent:]
    old = self.observations[:-keep_recent]
    
    # Résumé statistique des anciennes
    summary = self.summarize(old, compression_ratio)
    
    self.observations = recent
    self.summaries.append(summary)