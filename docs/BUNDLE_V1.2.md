# BUNDLE V1.2 — Modules Obligatoires
Vue d'Ensemble
Le Bundle V1.2 définit les 6 modules qui doivent être implémentés au-dessus du Kernel pour créer un SFL fonctionnel.

## Module 1 : Temporal Gradient Layer
Spécification
param_grad : Tensor[d_p, H]
horizons   : float[H]
H >= 1, configurable
Configuration V1.2 Default
H = 2
horizons = [1.0, 10.0]  # short, long

grad_short = gradient basé sur reward immédiat
grad_long  = gradient basé sur return discounté (TD(λ))
Interaction avec tau_rate
horizon_effective[h] = horizons[h] / tau_rate

Pattern accéléré (tau_rate > 1) → horizons effectifs plus courts
Pattern ralenti (tau_rate < 1) → horizons effectifs plus longs

## Module 2 : Structured Uncertainty Layer
Spécification
4 types d'incertitude + agrégation :

epistemic   : var(gradients entre critics)
aleatoric   : bruit estimé des observations
model       : incertitude structurelle
adversarial : score de suspicion de gaming
scalar      : combinaison pondérée
Implémentation V1.2 Minimale
epistemic = torch.var(torch.stack([c.grad for c in critics]), dim=0).mean()
aleatoric = 0.0  # stub, à raffiner
model = 0.0      # stub, à raffiner
adversarial = anomaly_score(reports)  # heuristique simple
scalar = epistemic  # ou moyenne pondérée

## Module 3 : Lineage Graph Layer
Spécification
DAG complet des relations entre patterns :

Nodes : patterns (nh_id)
Edges : 
  - SPAWNED_FROM (création)
  - FORKED_FROM (fork)
  - MERGED_INTO (merge)
  - INFLUENCED_BY (corrélation de gradients)
Opérations
class LineageGraph:
    def add_pattern(self, nh_id, parents: List[bytes32]) -> None
    def record_fork(self, parent: bytes32, child: bytes32) -> None
    def record_merge(self, parents: List[bytes32], child: bytes32) -> None
    def get_ancestors(self, nh_id, depth: int) -> List[bytes32]
    def get_descendants(self, nh_id, depth: int) -> List[bytes32]
    def compute_influence(self, source: bytes32, target: bytes32) -> float
Attribution de Crédit
Quand un pattern performe bien, crédit propagé aux ancêtres :
credit(ancestor) += decay^distance × performance(descendant)

## Module 4 : Economic Hooks Layer
Compute Hook
AllocSignal.tau_rate → Scheduler GPU

Mapping :
  tau_rate → priorité dans la queue
  tau_rate → multiplicateur de slots
  tau_rate → fréquence de scheduling
Capital Hook
ContributionScore + GradientPacket.uncertainty → Risk Engine

Usage :
  - Scoring pour allocation de capital
  - Pricing de risque
  - Sélection de projets
Compliance Tracking
Si scheduler ignore AllocSignal pendant > compliance_window :
  - Log violation
  - Appliquer non_compliance_penalty au prochain gradient
  - Augmenter uncertainty.adversarial du scheduler

## Module 5 : Field Memory Layer
Principe
Field Memory n'est pas un stockage passif.
C'est l'intégrale temporelle de Φ.

Stocker Report = ajouter contribution à ∫Φ dτ
Query = projection de l'intégrale sur une base
Structure
class FieldMemory:
    def integrate(self, report: Report, weight: float) -> None
        # field_integral += weight * embed(report)
    
    def influence_on_gradient(self, nh_id: bytes32) -> Tensor
        # Retourne comment l'historique influence le gradient présent
    
    def query(self, projection: Tensor, time_range: Tuple) -> Tensor
        # Projection de l'intégrale
Access Levels
Basé sur ContributionScore :

Level 0 : < 100 reports  → gradients de base seulement
Level 1 : 100-1k reports → gradients multi-horizon
Level 2 : 1k-10k reports → lineage graphs
Level 3 : > 10k reports  → données brutes du field memory

## Module 6 : Protocol Evolution Layer
Spécification
ProtocolState:
  version_base   : string        # "1.2.0"
  active_modules : bytes32[]     # IDs des modules actifs
  param_config   : Map<string, any>  # Configuration runtime

FieldKernelVariant:
  kernel_id      : bytes32
  F_params       : Tensor        # Paramètres de la dynamique
  adoption_count : uint64        # Patterns utilisant ce kernel
  performance    : float         # AGDP générée
Mécanisme de Sélection
1. Plusieurs FieldKernels peuvent coexister
2. Chaque pattern choisit son kernel
3. Kernels avec haute performance attirent plus de patterns
4. Kernels avec adoption < threshold après N cycles → éteints
Interface
class ProtocolEvolution:
    def register_variant(self, kernel: FieldKernelVariant) -> None
    def switch_kernel(self, nh_id: bytes32, new_kernel: bytes32) -> None
    def fork_kernel(self, base: bytes32, mutations: Tensor) -> bytes32
    def get_adoption_stats(self) -> Dict[bytes32, int]