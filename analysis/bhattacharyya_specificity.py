import math
from typing import Dict, Iterable, List, Tuple
from .base_strategy import AnalysisStrategy

class BhattacharyyaSpecificityStrategy(AnalysisStrategy):
    """
    Ranks neurons by how little they contribute to the global
    Bhattacharyya coefficient between the two groups.

    Let
        p₁[n] =   w₁[n] / Σₖ w₁[k]     (probability of neuron n in group1)
        p₂[n] =   w₂[n] / Σₖ w₂[k]

    • The global coefficient is BC = Σₙ √(p₁[n] · p₂[n]).
    • The per-neuron contribution is c[n] = √(p₁[n] · p₂[n]).

      A *smaller* c[n] ⇒ neuron is more group-specific.

    Sorting rule:
        1. ascending c[n]  (most specific first);
        2. tie-break by |p₁[n] − p₂[n]| (larger gap wins).

    The value returned in the tuple is the **signed** gap
    Δp = p₁[n] − p₂[n] so the steering code can keep its direction
    convention (positive ⇒ promotes group1, negative ⇒ promotes group2).
    """

    def get_name(self) -> str:
        return "bhattacharyya_specificity"

    @staticmethod
    def _normalise(vals: Dict[str, float]) -> Dict[str, float]:
        tot = sum(v for v in vals.values() if v > 0)
        return {k: max(0.0, v) / tot for k, v in vals.items()} if tot else {}

    def calculate_neuron_scores(
        self,
        valid_neurons: Iterable[str],
        weights: Dict[str, Dict[str, float]],
        counts:  Dict[str, Dict[str, int]],      
    ) -> List[Tuple[str, float]]:

        g1, g2 = list(weights.keys())            

        # Build *positive* weight dictionaries restricted to valid_neurons
        w1 = {n: max(0.0, weights[g1].get(n, 0.0)) for n in valid_neurons}
        w2 = {n: max(0.0, weights[g2].get(n, 0.0)) for n in valid_neurons}

        # Convert to probability distributions
        p1 = self._normalise(w1)
        p2 = self._normalise(w2)

        scored: List[Tuple[str, float, float]] = []   # (nid, signed_gap, contribution)

        for n in valid_neurons:
            bc_contrib = math.sqrt(p1.get(n, 0.0) * p2.get(n, 0.0))
            signed_gap = p1.get(n, 0.0) - p2.get(n, 0.0)
            scored.append((n, signed_gap, bc_contrib))

        # Sort: smallest contribution first, then by absolute gap
        scored.sort(key=lambda tpl: (tpl[2], -abs(tpl[1])))

        # Return (neuron_id, signed_gap) pairs in that order
        return [(nid, gap) for nid, gap, _ in scored]
