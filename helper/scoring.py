import math
from typing import Dict, List, Tuple, Set

class Scoring:
    """
    A helper class to handle different methods of scoring nodes and links
    and identifying critical sets based on those scores.
    """

    def __init__(self, critical_set_percentile: float):
        """
        Initializes the scoring helper.

        Args:
            critical_set_percentile: The percentage of top-scoring items to mark as "critical".
        """
        if not 0.0 <= critical_set_percentile <= 1.0:
            raise ValueError("critical_set_percentile must be between 0 and 1.")
        self.critical_set_percentile = critical_set_percentile

    def get_global_bhattacharyya_coefficient(self, dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
        """
        Calculates the global Bhattacharyya coefficient between two distributions.
        This measures the overall similarity of two activation patterns.

        Args:
            dist1: A dictionary representing the first distribution {event: value}.
            dist2: A dictionary representing the second distribution {event: value}.

        Returns:
            The Bhattacharyya coefficient, a value between 0 (no overlap) and 1 (identical).
        """
        all_keys = set(dist1.keys()) | set(dist2.keys())
        sum_of_products = 0.0
        for key in all_keys:
            val1 = dist1.get(key, 0.0)
            val2 = dist2.get(key, 0.0)
            sum_of_products += math.sqrt(val1 * val2)
        return sum_of_products

    def annotate_differential_scores(
        self,
        unified_nodes: List[Dict],
        unified_links: List[Dict],
        group1_name: str,
        group2_name: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Calculates a differential score based on the raw difference in activations and weights.
        Identifies critical nodes/links from the extremes (most positive and most negative differences).

        Args:
            unified_nodes: List of merged node data.
            unified_links: List of merged link data.
            group1_name: Name of the first group (e.g., 'male').
            group2_name: Name of the second group (e.g., 'female').

        Returns:
            A tuple containing the annotated nodes and links, now with 'is_critical' flags.
        """
        print("      --- Calculating Differential (Divergence) Scores ---")
        # Score links and nodes
        for link in unified_links:
            link['differential_score'] = abs(link['diff_weight'])

        for node in unified_nodes:
            activation1 = node.get(f'activation_{group1_name}', 0.0)
            activation2 = node.get(f'activation_{group2_name}', 0.0)
            node['signed_diff_activation'] = activation1 - activation2
            node['differential_score'] = abs(node['signed_diff_activation'])

        # Identify critical sets from positive and negative extremes
        sorted_nodes = sorted(unified_nodes, key=lambda x: x.get('signed_diff_activation', 0.0), reverse=True)
        sorted_links = sorted(unified_links, key=lambda x: x.get('diff_weight', 0.0), reverse=True)

        cutoff_nodes = max(1, int(len(sorted_nodes) * self.critical_set_percentile)) if sorted_nodes else 0
        cutoff_links = max(1, int(len(sorted_links) * self.critical_set_percentile)) if sorted_links else 0

        critical_node_ids = {n['node_id'] for n in sorted_nodes[:cutoff_nodes]} | \
                            {n['node_id'] for n in sorted_nodes[-cutoff_nodes:]}
        critical_link_ids = {(l['source'], l['target']) for l in sorted_links[:cutoff_links]} | \
                            {(l['source'], l['target']) for l in sorted_links[-cutoff_links:]}
        
        print(f"        - Identified {len(critical_node_ids)} critical divergent nodes and {len(critical_link_ids)} links.")

        # Annotate with the 'is_critical' flag
        for node in unified_nodes:
            node['is_critical'] = node['node_id'] in critical_node_ids
        for link in unified_links:
            link['is_critical'] = (link['source'], link['target']) in critical_link_ids

        return unified_nodes, unified_links

    def annotate_bhattacharyya_scores(
        self,
        unified_nodes: List[Dict],
        group1_name: str,
        group2_name: str
    ) -> List[Dict]:
        """
        Calculates a score for each node based on its contribution to the Bhattacharyya coefficient.
        This score represents shared importance or agreement. Identifies critical nodes that
        contribute the most to the similarity between the two groups.

        Args:
            unified_nodes: List of merged node data.
            group1_name: Name of the first group.
            group2_name: Name of the second group.

        Returns:
            The annotated list of nodes, now with 'is_bhattacharyya_critical' flags.
        """
        print("      --- Calculating Bhattacharyya (Convergence) Scores ---")
        if not unified_nodes:
            return []

        # Score each node by its contribution to the BC
        for node in unified_nodes:
            activation1 = node.get(f'activation_{group1_name}', 0.0)
            activation2 = node.get(f'activation_{group2_name}', 0.0)
            node['bhattacharyya_contribution'] = math.sqrt(activation1 * activation2)

        # Identify critical set from the top contributors
        sorted_nodes = sorted(unified_nodes, key=lambda x: x.get('bhattacharyya_contribution', 0.0), reverse=True)
        
        cutoff = max(1, int(len(sorted_nodes) * self.critical_set_percentile))
        critical_node_ids = {n['node_id'] for n in sorted_nodes[:cutoff]}
        
        print(f"        - Identified {len(critical_node_ids)} critical convergent nodes.")

        # Annotate with the 'is_bhattacharyya_critical' flag
        for node in unified_nodes:
            node['is_bhattacharyya_critical'] = node['node_id'] in critical_node_ids

        return unified_nodes