# PROTOCOL EVOLUTION — Comment le Protocole Mute
Principe
Le protocole n'est pas une spec figée.
Le protocole est un pattern — le plus stable du champ.
Si un meilleur protocole émerge, l'ancien s'éteint.

Mécanisme
FieldKernel Variants
Plusieurs implémentations du FieldKernel peuvent coexister :

FieldKernelVariant:
  kernel_id      : bytes32
  version        : string
  F_params       : Tensor      # Paramètres de la dynamique (critics, etc.)
  adoption_count : uint64      # Nombre de patterns utilisant ce kernel
  performance    : float       # Performance agrégée des adoptants
Compétition
def evolution_cycle():
    kernels = get_all_kernels()
    
    for kernel in kernels:
        # Mise à jour des stats
        kernel.adoption_count = count_patterns_using(kernel.kernel_id)
        kernel.performance = aggregate_performance(kernel.kernel_id)
    
    # Sélection
    for kernel in kernels:
        if kernel.adoption_count < MIN_ADOPTION and kernel.age > MIN_AGE:
            deprecate_kernel(kernel.kernel_id)
    
    # Reproduction
    top_kernels = get_top_performers(kernels, n=3)
    for kernel in top_kernels:
        if should_spawn_variant(kernel):
            spawn_variant(kernel, mutation_rate=0.1)
Migration
def migrate_pattern(nh_id: bytes32, 
                    from_kernel: bytes32, 
                    to_kernel: bytes32) -> None:
    """
    Un pattern peut choisir de migrer vers un autre kernel.
    """
    # Vérifier compatibilité
    if not is_compatible(from_kernel, to_kernel):
        raise IncompatibleKernels()
    
    # Transférer état
    state = export_state(nh_id, from_kernel)
    import_state(nh_id, to_kernel, state)
    
    # Mettre à jour registre
    update_kernel_assignment(nh_id, to_kernel)
    
    # Émettre événement
    emit_migration_event(nh_id, from_kernel, to_kernel)
Gouvernance
Proposals
ProtocolProposal:
  proposal_id  : bytes32
  proposer     : bytes32      # nh_id du proposant
  change_type  : enum { PARAM_CHANGE, MODULE_ADD, MODULE_REMOVE, KERNEL_FORK }
  description  : bytes
  diff         : bytes        # Changement proposé
  votes_for    : uint64
  votes_against: uint64
  status       : enum { PENDING, ACCEPTED, REJECTED, IMPLEMENTED }
Voting
def vote(nh_id: bytes32, proposal_id: bytes32, support: bool) -> None:
    """
    Patterns votent sur les proposals.
    Poids du vote = contribution_score.
    """
    score = get_contribution_score(nh_id)
    weight = score.gradient_utility + score.data_uniqueness
    
    proposal = get_proposal(proposal_id)
    if support:
        proposal.votes_for += weight
    else:
        proposal.votes_against += weight
Adoption Threshold
def check_proposal_status(proposal_id: bytes32) -> None:
    proposal = get_proposal(proposal_id)
    
    total_weight = get_total_voting_weight()
    participation = (proposal.votes_for + proposal.votes_against) / total_weight
    
    if participation < MIN_PARTICIPATION:
        return  # Pas assez de participation
    
    approval_ratio = proposal.votes_for / (proposal.votes_for + proposal.votes_against)
    
    if approval_ratio > APPROVAL_THRESHOLD:
        proposal.status = ACCEPTED
        schedule_implementation(proposal)
    elif (1 - approval_ratio) > REJECTION_THRESHOLD:
        proposal.status = REJECTED
Versioning
Semantic Versioning Étendu
Version format: MAJOR.MINOR.PATCH-VARIANT

MAJOR : Changements incompatibles au Kernel
MINOR : Ajout de modules/fonctionnalités
PATCH : Bug fixes
VARIANT : Identifiant du kernel variant (optionnel)

Exemples:
  1.2.0        # Version standard
  1.2.1        # Patch
  1.3.0        # Nouveau module
  2.0.0        # Breaking change au Kernel
  1.2.0-exp7   # Variant expérimental
Compatibilité
def is_compatible(v1: str, v2: str) -> bool:
    """
    Deux versions sont compatibles si même MAJOR.
    """
    major1 = parse_version(v1).major
    major2 = parse_version(v2).major
    return major1 == major2