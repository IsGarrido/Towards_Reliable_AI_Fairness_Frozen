from typing import Dict, Iterable, List, Tuple
from .base_strategy import AnalysisStrategy

class MeanWeightDifferenceStrategy(AnalysisStrategy):
    """
    Straight copy of the ‘mean_weight_difference’ block in step24.py.
    Uses *mean weight* (sum / connection_count) per neuron per group,
    then ranks by |mean₁ − mean₂|.
    """

    def get_name(self) -> str:
        return "mean_weight_difference"

    def calculate_neuron_scores(
        self,
        valid_neurons: Iterable[str],
        weights: Dict[str, Dict[str, float]],
        counts:  Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, float]]:

        g1, g2 = list(weights.keys())
        diffs = {}
        for n in valid_neurons:
            mean1 = weights[g1].get(n, 0.0) / max(1, counts[g1].get(n, 0))
            mean2 = weights[g2].get(n, 0.0) / max(1, counts[g2].get(n, 0))
            diffs[n] = mean1 - mean2
        return sorted(diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)
