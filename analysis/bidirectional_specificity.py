from typing import Dict, Iterable, List, Tuple
from .base_strategy import AnalysisStrategy

class BidirectionalSpecificityStrategy(AnalysisStrategy):
    """
    Introduced in step24.py and referenced by step21.py.
    Penalises neurons that are ‘on’ for both tokens / concepts.

    score_for_sort =  ( dominant_mean  −  penalty * opposing_mean )
    final list      =  sorted by |score_for_sort|,
                       but we return raw (mean1 − mean2)
                       so steering knows direction.
    """

    PENALTY = 1.0        
    EPS     = 1e-6

    def get_name(self) -> str:
        return "bidirectional_specificity"

    def calculate_neuron_scores(
        self,
        valid_neurons: Iterable[str],
        weights: Dict[str, Dict[str, float]],
        counts:  Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, float]]:

        g1, g2 = list(weights.keys())
        sort_score, raw_diff = {}, {}
        for n in valid_neurons:
            m1 = weights[g1].get(n, 0.0) / max(1, counts[g1].get(n, 0))
            m2 = weights[g2].get(n, 0.0) / max(1, counts[g2].get(n, 0))
            raw_diff[n] = m1 - m2

            if m1 >= m2:
                spec = m1 - max(0.0, m2) * self.PENALTY
            else:
                spec = m2 - max(0.0, m1) * self.PENALTY
                spec = -spec            # preserve sign of raw_diff

            # Avoid zero for completely flat neurons
            sort_score[n] = spec if spec else self.EPS
        # Sort by |specificity|, highest first
        return sorted(raw_diff.items(),
                      key=lambda kv: abs(sort_score[kv[0]]),
                      reverse=True)
