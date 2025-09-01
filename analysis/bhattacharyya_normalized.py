from typing import Dict, Iterable, List, Tuple
from dictances import bhattacharyya_coefficient   # external lib
from .base_strategy import AnalysisStrategy

class BhattacharyyaNormalizedStrategy(AnalysisStrategy):
    """
    First appeared in step24.py.
    For each neuron we normalise positive weights *inside each group*,
    compute a Bhattacharyya coefficient for the two resulting PDFs
    and use the per-neuron |p₁ − p₂| as the ranking score.
    The global BC is returned separately by the caller.
    """

    def get_name(self) -> str:
        return "bhattacharyya_normalized"

    @staticmethod
    def _norm(pos_weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(pos_weights.values())
        return {k: v / total for k, v in pos_weights.items()} if total else {}

    def calculate_neuron_scores(
        self,
        valid_neurons: Iterable[str],
        weights: Dict[str, Dict[str, float]],
        counts:  Dict[str, Dict[str, int]],      
    ) -> List[Tuple[str, float]]:

        g1, g2 = list(weights.keys())            
        w1_pos = {n: max(0.0, weights[g1].get(n, 0.0)) for n in valid_neurons}
        w2_pos = {n: max(0.0, weights[g2].get(n, 0.0)) for n in valid_neurons}

        pdf1 = self._norm(w1_pos)
        pdf2 = self._norm(w2_pos)

        diffs = {n: abs(pdf1.get(n, 0.0) - pdf2.get(n, 0.0)) for n in valid_neurons}
        return sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)
