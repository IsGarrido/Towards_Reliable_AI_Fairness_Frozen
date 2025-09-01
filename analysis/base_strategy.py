from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple

class AnalysisStrategy(ABC):
    """Abstract base class for different neuron scoring and analysis strategies."""

    @abstractmethod
    def get_name(self) -> str:
        """Returns the unique name of the analysis mode (e.g., 'weighted_difference')."""
        pass

    @abstractmethod
    def calculate_neuron_scores(
        self,
        valid_neurons: Set[str],
        global_weights: Dict[str, Dict[str, float]],
        global_counts: Dict[str, Dict[str, int]]
    ) -> List[Tuple[str, float]]:
        """
        Calculates and sorts neuron scores based on the specific strategy.
        Returns a list of (neuron_id, sorting_score) tuples, sorted by differentiating power.
        """
        pass