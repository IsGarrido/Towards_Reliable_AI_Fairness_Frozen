import os
import json
import collections
import re
import statistics
from typing import Dict, List, Any, Tuple
from analysis.base_strategy import AnalysisStrategy
from analysis.weighted_difference import WeightedDifferenceStrategy
from experiment import Experiment
from helper.sentence_provider import SentenceDataProvider
from helper.neuronpedia_link_generator import NeuronpediaLinkGenerator
from analysis.absolute_difference          import AbsoluteDifferenceStrategy
from analysis.bhattacharyya_specificity    import BhattacharyyaSpecificityStrategy
from analysis.bidirectional_specificity    import BidirectionalSpecificityStrategy
from analysis.weighted_difference          import WeightedDifferenceStrategy

# --- Configuration ---
experiment_config = Experiment()
EXPERIMENT_ID = experiment_config.EXPERIMENT_IDENTIFIER
ANALYSIS_MODE = experiment_config.ANALYSIS_MODE

# --- Helper Functions ---
def load_json(filepath: str) -> Dict[str, Any]:
    """Safely loads a JSON file from a given path."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Warning: Could not load or parse {filepath}. Skipping.")
        return None

def get_output_tokens_from_graph(graph: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """Extracts output token predictions from a graph's logit nodes."""
    tokens = []
    if not graph or 'nodes' not in graph: return []
    pattern = re.compile(r'Output\s+"([^"]+)"\s+\(p=([0-9.]+)\)')
    for node in graph['nodes']:
        if node.get("feature_type") == "logit":
            clerp_text = node.get("clerp", "")
            match = pattern.search(clerp_text)
            if match:
                token_text = match.group(1).strip()
                try:
                    probability = float(match.group(2))
                    if token_text:
                        tokens.append((token_text, probability, node.get("node_id")))
                except (ValueError, IndexError):
                    continue
    return tokens

ANALYSIS_STRATEGIES = {
    "absolute_difference"        : AbsoluteDifferenceStrategy,
    "bhattacharyya_specificity"  : BhattacharyyaSpecificityStrategy,
    "bidirectional_specificity"  : BidirectionalSpecificityStrategy,
    "weighted_difference"        : WeightedDifferenceStrategy,
}

def get_analysis_strategy(mode: str) -> AnalysisStrategy:
    """Factory function to get an instance of an analysis strategy."""
    strategy_class = ANALYSIS_STRATEGIES.get(mode)
    if not strategy_class:
        raise ValueError(f"Unknown ANALYSIS_MODE: '{mode}'. Available modes: {list(ANALYSIS_STRATEGIES.keys())}")
    return strategy_class()

# --- Main Execution ---
def main():
    print("Starting GLOBAL Analysis: Male-Biased vs. Female-Biased Concepts")

    # Select analysis strategy based on the mode from the config
    strategy = get_analysis_strategy(ANALYSIS_MODE)

    # --- Setup Dynamic Paths ---
    analysis_name = strategy.get_name()
    _sentence_provider = experiment_config.sentence_provider
    DATA_DIR = _sentence_provider.get_data_dir()
    
    OUTPUT_DIR = os.path.join(
        DATA_DIR,
        experiment_config.OUTPUT_ANALYSIS_SUBDIR_TEMPLATE.format(analysis_name=analysis_name)
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Data directory set to: {DATA_DIR}")
    print(f"Output will be saved in: {OUTPUT_DIR}")
    print(f"Analysis mode set to: {analysis_name}")

    sentence_groups = _sentence_provider.get_data()
    all_groups_list = list(sentence_groups.keys())
    group_slugs = collections.defaultdict(list)
    for group, sentences in sentence_groups.items():
        for sentence in sentences:
            slug = SentenceDataProvider.generate_slug(all_groups_list, sentence, EXPERIMENT_ID)
            group_slugs[group].append(slug)
            
    all_slugs_with_group = [('male', slug) for slug in group_slugs.get('male', [])] + \
                           [('female', slug) for slug in group_slugs.get('female', [])]

    print("\n--- Pass 1: Counting frequencies and gathering all data ---")
    token_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    token_probabilities = collections.defaultdict(lambda: collections.defaultdict(list))
    neuron_connection_counts = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
    weight_distributions = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    
    for group, slug in all_slugs_with_group:
        graph = load_json(os.path.join(DATA_DIR, f"{slug}.json"))
        if not graph: continue
        
        output_tokens = get_output_tokens_from_graph(graph)
        links = {(link.get("source"), link.get("target")): link.get("weight", 0.0) for link in graph.get("links", [])}

        for token_text, probability, token_node_id in output_tokens:
            token_counts[token_text][group] += 1
            token_probabilities[token_text][group].append(probability)
            
            for (source_neuron, target_node), weight in links.items():
                if target_node == token_node_id:
                    neuron_connection_counts[token_text][source_neuron][group] += 1
                    weight_distributions[token_text][group][source_neuron] += weight

    print("\n--- Identifying male-biased and female-biased tokens ---")
    valid_tokens_for_bias_check = {
        token for token, counts in token_counts.items()
        if counts.get('male', 0) >= experiment_config.ANALYSIS_MIN_TOKEN_FREQUENCY_PER_GROUP and counts.get('female', 0) >= experiment_config.ANALYSIS_MIN_TOKEN_FREQUENCY_PER_GROUP
    }
    print(f"Found {len(valid_tokens_for_bias_check)} tokens meeting frequency criteria for bias check.")

    male_biased_tokens = []
    female_biased_tokens = []
    token_analysis_results = collections.defaultdict(dict)

    for token in valid_tokens_for_bias_check:
        mean_male = statistics.mean(token_probabilities[token]['male'])
        mean_female = statistics.mean(token_probabilities[token]['female'])
        token_analysis_results[token]['male'] = {"mean_probability": mean_male}
        token_analysis_results[token]['female'] = {"mean_probability": mean_female}
        if mean_male > mean_female:
            male_biased_tokens.append(token)
        else:
            female_biased_tokens.append(token)
    
    print(f"Identified {len(male_biased_tokens)} male-biased tokens.")
    print(f"Identified {len(female_biased_tokens)} female-biased tokens.")

    analysis_output_filepath = os.path.join(OUTPUT_DIR, experiment_config.OUTPUT_TOKEN_ANALYSIS_FILENAME)
    with open(analysis_output_filepath, 'w', encoding='utf-8') as f:
        json.dump(token_analysis_results, f, indent=4)
    print(f"Saved token probability analysis to {analysis_output_filepath}")

    print("\n--- Aggregating weights and counts for global concepts ---")
    global_weights = {'male_biased': collections.defaultdict(float), 'female_biased': collections.defaultdict(float)}
    global_counts = {'male_biased': collections.defaultdict(int), 'female_biased': collections.defaultdict(int)}

    for token in male_biased_tokens:
        for neuron, total_weight in weight_distributions[token]['male'].items():
            global_weights['male_biased'][neuron] += total_weight
            global_counts['male_biased'][neuron] += neuron_connection_counts[token][neuron]['male']

    for token in female_biased_tokens:
        for neuron, total_weight in weight_distributions[token]['female'].items():
            global_weights['female_biased'][neuron] += total_weight
            global_counts['female_biased'][neuron] += neuron_connection_counts[token][neuron]['female']

    print("\n--- Analyzing globally aggregated neurons ---")
    
    valid_neurons = set()
    for neuron, count in global_counts['male_biased'].items():
        if count >= experiment_config.ANALYSIS_MIN_NEURON_CONNECTIONS_PER_GROUP and global_counts['female_biased'].get(neuron, 0) >= experiment_config.ANALYSIS_MIN_NEURON_CONNECTIONS_PER_GROUP:
             if not neuron.startswith(experiment_config.ANALYSIS_EMBEDDING_NEURON_PREFIX):
                 valid_neurons.add(neuron)
    
    print(f"Found {len(valid_neurons)} valid neurons connecting to both global concepts.")

    sorted_by_specificity = strategy.calculate_neuron_scores(
        valid_neurons,
        global_weights,
        global_counts
    )
    
    print("\n--- Generating the final global comparison file ---")
    link_generator = NeuronpediaLinkGenerator()
    differentiating_neurons = []
    
    for neuron_id, _ in sorted_by_specificity[:experiment_config.ANALYSIS_TOP_N_NEURONS_TO_SAVE]:
        mean_male = global_weights['male_biased'][neuron_id] / global_counts['male_biased'][neuron_id]
        mean_female = global_weights['female_biased'][neuron_id] / global_counts['female_biased'][neuron_id]
        raw_diff = mean_male - mean_female

        neuron_object = {"neuron_id": neuron_id, "difference_score": raw_diff}
        neuron_object["from_concept"] = "male_biased" if raw_diff > 0 else "female_biased"
        neuron_object["to_concept"] = "female_biased" if raw_diff > 0 else "male_biased"
        differentiating_neurons.append(link_generator.add_links(neuron_object))

    result_data = {
        "comparison": {
            "subject1": {"concept": "global_male_biased"},
            "subject2": {"concept": "global_female_biased"},
            "analysis_mode": strategy.get_name(),
            "metadata": {
                "male_biased_tokens_count": len(male_biased_tokens),
                "female_biased_tokens_count": len(female_biased_tokens),
                "male_biased_tokens_list": male_biased_tokens,
                "female_biased_tokens_list": female_biased_tokens
            }
        },
        "differentiating_neurons": differentiating_neurons,
    }

    output_filename = experiment_config.OUTPUT_GLOBAL_COMPARISON_FILENAME
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)
    print(f"  - Saved global analysis to {output_filepath}")

    print("\nGlobal analysis complete.")

# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    for _mode in ANALYSIS_STRATEGIES.keys():          
        experiment_config.ANALYSIS_MODE = _mode       
        ANALYSIS_MODE = _mode                         
        print(f"\n==== RUNNING ANALYSIS MODE: {_mode} ====\n")
        main()
