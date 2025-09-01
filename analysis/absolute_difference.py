from typing import Dict, Iterable, List, Tuple
from .base_strategy import AnalysisStrategy

class AbsoluteDifferenceStrategy(AnalysisStrategy):
    """
    Captures the simpler logic from step22.py (top-105 selection).

    score = |Pr₁ − Pr₂|
    """

    def get_name(self) -> str:
        return "absolute_difference"

    def calculate_neuron_scores(
        self,
        valid_neurons: Iterable[str],
        weights: Dict[str, Dict[str, float]],
        counts:  Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, float]]:

        g1, g2 = list(weights.keys())
        diffs = {n: abs(weights[g1].get(n, 0.0) - weights[g2].get(n, 0.0))
                 for n in valid_neurons}
        return sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)
