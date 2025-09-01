import os
import json
import itertools
from typing import List, Dict, Tuple, Optional

try:
    from experiment import Experiment
    from scipy.stats import spearmanr
except ImportError as e:
    print(f"Error: Could not import necessary libraries. {e}")
    print("Please ensure you have 'scipy' installed (`pip install scipy`) and that the script can access the 'experiment' module.")
    exit()

ANALYSIS_MODES = Experiment.ANALYSIS_MODES
TOP_N = 50

def load_analysis_results(analysis_name: str, config: Experiment) -> Optional[List[Tuple[str, float]]]:
    """
    Loads the ranked list of differentiating neurons for a given analysis mode.
    """
    try:
        _sentence_provider = config.sentence_provider
        data_dir = _sentence_provider.get_data_dir()
        
        output_subdir = config.OUTPUT_ANALYSIS_SUBDIR_TEMPLATE.format(analysis_name=analysis_name)
        filename = config.OUTPUT_GLOBAL_COMPARISON_FILENAME
        filepath = os.path.join(data_dir, output_subdir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        differentiating_neurons = data.get("differentiating_neurons", [])
        
        ranked_list = [
            (neuron.get("neuron_id"), neuron.get("difference_score"))
            for neuron in differentiating_neurons if neuron.get("neuron_id")
        ]
        return ranked_list

    except FileNotFoundError:
        print(f"Warning: Results file not found for '{analysis_name}'. Skipping.")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not parse results for '{analysis_name}'. Error: {e}. Skipping.")
        return None


def compare_rankings(list_a: List[Tuple[str, float]], list_b: List[Tuple[str, float]], top_n: int) -> Dict:
    """
    Compares two ranked lists using Top-N overlap and Spearman's correlation.
    """
    # 1. Top-N Overlap Calculation
    top_n_a = {neuron_id for neuron_id, score in list_a[:top_n]}
    top_n_b = {neuron_id for neuron_id, score in list_b[:top_n]}
    
    overlap_count = len(top_n_a.intersection(top_n_b))
    overlap_percentage = (overlap_count / top_n) * 100 if top_n > 0 else 0

    # 2. Spearman's Rank Correlation
    ranks_a = {neuron_id: i for i, (neuron_id, score) in enumerate(list_a)}
    ranks_b = {neuron_id: i for i, (neuron_id, score) in enumerate(list_b)}

    common_neurons = list(ranks_a.keys() & ranks_b.keys())
    
    if len(common_neurons) < 2:
        # Not enough common data to compute correlation
        correlation, p_value = (0.0, 1.0)
    else:
        rank_vec_a = [ranks_a[neuron_id] for neuron_id in common_neurons]
        rank_vec_b = [ranks_b[neuron_id] for neuron_id in common_neurons]
        correlation, p_value = spearmanr(rank_vec_a, rank_vec_b)

    return {
        "top_n_overlap_count": overlap_count,
        "top_n_overlap_percentage": overlap_percentage,
        "spearman_rho": correlation
    }


# --- Main Execution ---

def main():
    """
    Main function to load all analysis results and compare them pair-wise.
    """
    print("--- Starting Analysis Strategy Comparison ---")
    
    try:
        experiment_config = Experiment()
        print(f"Loaded config for experiment: '{experiment_config.EXPERIMENT_IDENTIFIER}'")
    except Exception as e:
        print(f"Error: Could not initialize Experiment configuration. {e}")
        return

    # 1. Load all available analysis results
    all_results = {}
    for mode in ANALYSIS_MODES:
        result = load_analysis_results(mode, experiment_config)
        if result:
            all_results[mode] = result
            print(f"  Successfully loaded {len(result)} neurons for '{mode}'.")

    if len(all_results) < 2:
        print("\nError: Need at least two valid analysis results to compare. Exiting.")
        return

    # 2. Get all unique pairs of strategies to compare
    strategy_pairs = list(itertools.combinations(all_results.keys(), 2))
    print(f"\nFound {len(strategy_pairs)} pairs of strategies to compare.")

    # 3. Perform and print comparisons
    print("\n--- Comparison Report ---")
    for strategy_a, strategy_b in strategy_pairs:
        print(f"\n--------------------------------------------------")
        print(f"📊 Comparing '{strategy_a}' vs. '{strategy_b}'")
        print(f"--------------------------------------------------")

        list_a = all_results[strategy_a]
        list_b = all_results[strategy_b]

        comparison_stats = compare_rankings(list_a, list_b, TOP_N)

        # Print Top-N Overlap results
        print(f"  Overlap in Top {TOP_N}:")
        print(f"    - Common Neurons: {comparison_stats['top_n_overlap_count']}/{TOP_N}")
        print(f"    - Similarity: {comparison_stats['top_n_overlap_percentage']:.2f}%")

        # Print Spearman's Rho results
        rho = comparison_stats['spearman_rho']
        print(f"\n  Spearman's Rank Correlation (ρ):")
        print(f"    - Rho Value: {rho:.4f}")
        
        # Add a qualitative verdict based on the rho value
        if rho > 0.9:
            verdict = "Verdict: 🟢 Highly Redundant. The rankings are nearly identical."
        elif rho > 0.7:
            verdict = "Verdict: 🟡 Very Similar. They prioritize neurons in a similar order."
        elif rho > 0.4:
            verdict = "Verdict: 🔵 Moderately Similar. They share some ranking trends."
        else:
            verdict = "Verdict: 🔴 Dissimilar. The strategies capture different phenomena."
        print(f"    - {verdict}")

    print("\n\n--- Comparison Complete ---")


if __name__ == "__main__":
    main()