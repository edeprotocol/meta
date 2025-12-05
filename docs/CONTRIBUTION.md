# CONTRIBUTION — Skin in the Game Synthétique
Principe
Pas de free-riding dans le Field.
Tu contribues → tu accèdes.
Tu ne contribues pas → tu restes au niveau 0.

ContributionScore
Composantes
ContributionScore:
  reports_submitted : uint64   # Volume brut
  gradient_utility  : float    # Qualité des contributions
  data_uniqueness   : float    # Nouveauté informationnelle
  access_level      : uint8    # Niveau résultant (0-3)
Calcul
def compute_contribution_score(nh_id: bytes32) -> ContributionScore:
    reports = get_reports_by(nh_id)
    
    # Volume
    reports_submitted = len(reports)
    
    # Utilité : combien les gradients basés sur ces données ont aidé
    gradient_utility = compute_gradient_utility(reports)
    
    # Unicité : entropie relative vs pool global
    data_uniqueness = compute_uniqueness(reports)
    
    # Niveau d'accès
    access_level = compute_access_level(
        reports_submitted, 
        gradient_utility, 
        data_uniqueness
    )
    
    return ContributionScore(
        reports_submitted=reports_submitted,
        gradient_utility=gradient_utility,
        data_uniqueness=data_uniqueness,
        access_level=access_level
    )
Gradient Utility
def compute_gradient_utility(reports: List[Report]) -> float:
    """
    Mesure combien les gradients entraînés sur ces reports
    ont amélioré la performance d'autres patterns.
    """
    utility = 0.0
    for report in reports:
        # Gradients qui ont utilisé ce report
        derived_gradients = get_gradients_using(report)
        for grad in derived_gradients:
            # Performance du pattern après avoir reçu ce gradient
            perf_delta = get_performance_delta(grad.nh_id, grad.timestamp)
            utility += perf_delta
    
    return utility / len(reports) if reports else 0.0
Data Uniqueness
def compute_uniqueness(reports: List[Report]) -> float:
    """
    Entropie relative des contributions vs pool global.
    Haut = contributions apportent de l'information nouvelle.
    Bas = contributions redondantes.
    """
    report_embeddings = [embed(r) for r in reports]
    global_distribution = get_global_embedding_distribution()
    
    kl_divergence = compute_kl(report_embeddings, global_distribution)
    
    # Normaliser entre 0 et 1
    return sigmoid(kl_divergence)
Access Levels
Seuils
def compute_access_level(reports: int, utility: float, uniqueness: float) -> int:
    score = (
        0.4 * normalize(reports, max=10000) +
        0.4 * utility +
        0.2 * uniqueness
    )
    
    if score < 0.1:
        return 0
    elif score < 0.3:
        return 1
    elif score < 0.6:
        return 2
    else:
        return 3
Droits par Niveau
Level 0:
  - Recevoir gradients de base (short horizon, scalar uncertainty)
  - Pas d'accès à Field Memory
  - Pas de lineage info

Level 1:
  - Gradients multi-horizon
  - Uncertainty structurée
  - Statistiques agrégées de Field Memory

Level 2:
  - Tout niveau 1
  - Accès aux lineage graphs
  - Query sur patterns similaires

Level 3:
  - Accès complet
  - Export de données
  - Participation aux décisions de protocole
Incentives
Récompenses
def distribute_rewards(cycle: int) -> None:
    """
    Chaque cycle, les top contributeurs reçoivent des bonus.
    """
    scores = get_all_contribution_scores()
    top_10_percent = get_top_percentile(scores, 0.1)
    
    for nh_id in top_10_percent:
        # Bonus sur tau_rate
        current_tau = get_tau_rate(nh_id)
        set_tau_rate(nh_id, current_tau * 1.1)
        
        # Bonus sur access
        promote_access_level(nh_id)
Pénalités
def apply_penalties(cycle: int) -> None:
    """
    Patterns qui consomment sans contribuer sont pénalisés.
    """
    for nh_id in get_all_patterns():
        consumption = get_gradient_consumption(nh_id)
        contribution = get_contribution_score(nh_id)
        
        ratio = contribution.gradient_utility / (consumption + 1e-6)
        
        if ratio < 0.1:  # Consomme 10x plus qu'il ne contribue
            current_tau = get_tau_rate(nh_id)
            set_tau_rate(nh_id, current_tau * 0.9)