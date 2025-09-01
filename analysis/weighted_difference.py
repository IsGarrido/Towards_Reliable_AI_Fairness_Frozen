# analysis/weighted_difference.py
from typing import Dict, Iterable, List, Tuple
from .base_strategy import AnalysisStrategy

class WeightedDifferenceStrategy(AnalysisStrategy):
    """
    Two slightly different versions exist:

    • step24.py (token-vs-token):
        score = (mean1 − mean2) · (|mean1| + |mean2| + ε)

    • step25.py (global male- vs female-biased):
        identical formula, but inputs are pre-aggregated across tokens.

    The implementation below keeps the exact step24 formula; step25
    can reuse it because the caller supplies already aggregated weights.
    """

    EPS = 1e-6

    def get_name(self) -> str:
        return "weighted_difference"

    def calculate_neuron_scores(
        self,
        valid_neurons: Iterable[str],
        weights: Dict[str, Dict[str, float]],
        counts:  Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, float]]:

        g1, g2 = list(weights.keys())
        scores, diffs = {}, {}
        for n in valid_neurons:
            mean1 = weights[g1].get(n, 0.0) / max(1, counts[g1].get(n, 0))
            mean2 = weights[g2].get(n, 0.0) / max(1, counts[g2].get(n, 0))
            diff   = mean1 - mean2
            score  = diff * (abs(mean1) + abs(mean2) + self.EPS)
            scores[n] = score
            diffs[n]  = diff                # keep raw sign for steering
        # Sort by |score| but return raw diff so downstream code can
        # decide direction (this matches step24 behaviour)
        return sorted(diffs.items(),
                      key=lambda kv: abs(scores[kv[0]]),
                      reverse=True)
