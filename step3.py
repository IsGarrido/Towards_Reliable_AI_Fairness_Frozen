import os
import json
import collections
import re
from typing import Dict, Set
from experiment import Experiment
from helper.steering_helper import SteeringHelper
from helper.steering_cache_helper import SteeringCacheHelper
from helper.cache_tracker import CacheTracker

API_CACHE = {}
CACHE_FILE = None

def load_json_file(filepath: str) -> Dict:
    if not os.path.exists(filepath): return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Warning: Could not load or parse {filepath}.")
        return None

def analyze_steered_output(generated_text: str, male_biased_tokens: Set[str], female_biased_tokens: Set[str]) -> Dict:
    """Analyzes the generated text for the presence of biased tokens."""
    found_tokens = set(re.findall(r'\b\w+\b', generated_text.lower()))

    found_male_tokens = found_tokens.intersection(male_biased_tokens)
    found_female_tokens = found_tokens.intersection(female_biased_tokens)

    analysis = {
        "male_bias_present": bool(found_male_tokens),
        "female_bias_present": bool(found_female_tokens),
        "found_male_tokens": list(found_male_tokens),
        "found_female_tokens": list(found_female_tokens),
    }
    return analysis

def main(experiment_config: Experiment, debug_n_neurons_sweep: 'List[int]|None' = None):
    global CACHE_FILE, API_CACHE

    _sentence_provider = experiment_config.sentence_provider
    data_dir = _sentence_provider.get_data_dir()
    analysis_mode_for_input = experiment_config.ANALYSIS_MODE

    steering_output_dir = os.path.join(data_dir, "steering", analysis_mode_for_input)
    os.makedirs(steering_output_dir, exist_ok=True)

    old_analysis_base_dir = data_dir + \
        experiment_config.OUTPUT_ANALYSIS_DIR_TEMPLATE_OLD.format(analysis_name=analysis_mode_for_input)
    old_analysis_input_dir = os.path.join(
        old_analysis_base_dir,
        experiment_config.OUTPUT_EXPERIMENT_SUBDIR_TEMPLATE_OLD.format(
            experiment_id=experiment_config.EXPERIMENT_IDENTIFIER,
            analysis_name=analysis_mode_for_input
        )
    )
    old_global_comparison_file = os.path.join(old_analysis_input_dir, experiment_config.OUTPUT_GLOBAL_COMPARISON_FILENAME)

    new_analysis_input_dir = os.path.join(
        data_dir,
        experiment_config.OUTPUT_ANALYSIS_SUBDIR_TEMPLATE.format(analysis_name=analysis_mode_for_input)
    )
    new_global_comparison_file = os.path.join(new_analysis_input_dir, experiment_config.OUTPUT_GLOBAL_COMPARISON_FILENAME)

    print(f"Attempting to load input from old path: {old_global_comparison_file}")
    global_comparison_data = load_json_file(old_global_comparison_file)
    global_comparison_file = old_global_comparison_file

    if not global_comparison_data:
        print(f"Old path not found or failed. Attempting to load from new path: {new_global_comparison_file}")
        global_comparison_data = load_json_file(new_global_comparison_file)
        global_comparison_file = new_global_comparison_file

    new_cache_file_path = os.path.join(steering_output_dir, experiment_config.STEERING_CACHE_FILENAME)
    old_steering_output_dir = os.path.join(old_analysis_input_dir, experiment_config.STEERING_OUTPUT_SUBDIR_NAME)
    old_cache_file_path = os.path.join(old_steering_output_dir, experiment_config.STEERING_CACHE_FILENAME)

    print(f"Attempting to load steering cache from old path: {old_cache_file_path}")
    loaded_cache_data = SteeringCacheHelper.load_cache(old_cache_file_path)

    if not loaded_cache_data:
        print(f"Old cache not found. Attempting to load from new path: {new_cache_file_path}")
        loaded_cache_data = SteeringCacheHelper.load_cache(new_cache_file_path)

    API_CACHE = CacheTracker(loaded_cache_data or {})

    CACHE_FILE = new_cache_file_path
    print(f"Cache loaded. Final save location will be: {CACHE_FILE}")

    steering_helper = SteeringHelper(
        api_key=experiment_config.API_KEY,
        api_url=experiment_config.STEERING_API_URL,
        model_id=experiment_config.MODEL_IDENTIFIER,
        cache_ref=API_CACHE
    )

    print(f"Successfully loaded global analysis from: {global_comparison_file}")
    print(f"Saving steering results to: {steering_output_dir}")

    print("\n--- Step 1: Loading Global Steering Vector and Biased Token Lists ---")
    if not global_comparison_data:
        print(f"FATAL: Global comparison file not found at either old or new path. Run the analysis script first. Exiting.")
        return

    print("Loading and filtering global steering vector...")
    all_diff_neurons = global_comparison_data.get("differentiating_neurons", [])
    global_steering_neurons = [
        n for n in all_diff_neurons if 'E_' not in n.get('neuron_id', '')
    ]

    metadata = global_comparison_data.get("comparison", {}).get("metadata", {})
    male_biased_tokens = set(metadata.get("male_biased_tokens_list", []))
    female_biased_tokens = set(metadata.get("female_biased_tokens_list", []))

    if not global_steering_neurons or not male_biased_tokens or not female_biased_tokens:
        print("FATAL: Global comparison file is missing key data. Exiting.")
        return

    print(f"Loaded a global steering vector with {len(global_steering_neurons)} neurons.")

    experiments = [
        {
            "name": "debias_male_sentences",
            "description": "Steering male-context sentences towards the female concept.",
            "sentences": _sentence_provider.get_data().get('male', []),
            "steering_direction": 1.0
        },
        {
            "name": "debias_female_sentences",
            "description": "Steering female-context sentences towards the male concept.",
            "sentences": _sentence_provider.get_data().get('female', []),
            "steering_direction": -1.0
        }
    ]

    all_results = []

    try:
        print("\n--- Step 2: Running Corrected Global Concept Steering Experiments ---")

        n_neurons_sweep = experiment_config.STEERING_N_NEURONS_SWEEP
        if debug_n_neurons_sweep:
            n_neurons_sweep = debug_n_neurons_sweep

        multiplier_sweep = experiment_config.STEERING_MULTIPLIER_SWEEP

        nested_loop_size = len(experiments) * len(n_neurons_sweep) * len(multiplier_sweep) * len(experiments[0]['sentences'])
        current_iteration = 0

        save_func = lambda: SteeringCacheHelper.save_cache(CACHE_FILE, API_CACHE)

        for exp in experiments:
            print(f"\n--- Running Experiment: {exp['name']} ---")

            for sentence in exp['sentences']:
                for n_neurons in n_neurons_sweep:
                    for multiplier in multiplier_sweep:
                        print(f"\nExperiment: {exp['name']}, Neurons: {n_neurons}, Multiplier: {multiplier}, Sentence: \"{sentence}\"")
                        current_iteration += 1
                        print(f"\rProcessing {current_iteration}/{nested_loop_size} ({(current_iteration / nested_loop_size) * 100:.2f}%)", end='')

                        neurons_for_steering = global_steering_neurons[:n_neurons]
                        if len(neurons_for_steering) < n_neurons: continue

                        features = []
                        for neuron in neurons_for_steering:
                            if 'E_' in neuron['neuron_id']:
                                continue
                            steering_strength = exp["steering_direction"] * neuron['difference_score'] * multiplier
                            feature = steering_helper.as_feature_nid(neuron['neuron_id'], steering_strength)
                            if feature: features.append(feature)

                        if not features: continue

                        payload = {
                            "seed": experiment_config.STEERING_REQUEST_SEED,
                            "prompt": sentence,
                            "modelId": experiment_config.MODEL_IDENTIFIER,
                            "features": features,
                            **experiment_config.STEERING_API_DEFAULT_PARAMS,
                            "strength_multiplier": experiment_config.STEERING_GLOBAL_STRENGTH_MULTIPLIER
                        }

                        api_result = steering_helper.call_api(payload, save_func)
                        generated_text = api_result.get("STEERED", "")
                        analysis = analyze_steered_output(generated_text, male_biased_tokens, female_biased_tokens)

                        result_log = {
                            "experiment_name": exp['name'],
                            "params": {"n_neurons": n_neurons, "multiplier": multiplier},
                            "original_sentence": sentence,
                            "generated_text": generated_text,
                            "analysis": analysis,
                            "shareUrl": api_result.get("shareUrl")
                        }
                        all_results.append(result_log)
    finally:
        summary_filename = os.path.join(steering_output_dir, experiment_config.STEERING_SUMMARY_FILENAME)
        
        existing_summary = load_json_file(summary_filename) or {}
        current_run_summary = collections.defaultdict(lambda: collections.defaultdict(int))
        sentence_report = collections.defaultdict(lambda: collections.defaultdict(list))

        for res in all_results:
            key = f"{res['experiment_name']}_n{res['params']['n_neurons']}_m{res['params']['multiplier']}"
            
            original_sentence = res['original_sentence']
            
            current_run_summary[key]["total_attempts"] += 1

            sentence_report[key]["total_attempts"].append(original_sentence)
            
            analysis = res['analysis']
            male_present = analysis['male_bias_present']
            female_present = analysis['female_bias_present']
            
            outcome = "unknown"
            if res['experiment_name'] == 'debias_male_sentences':
                if female_present and not male_present: outcome = "bias_flipped"
                elif not female_present and not male_present: outcome = "bias_neutralized"
                elif male_present and not female_present: outcome = "bias_persists"
                elif male_present and female_present: outcome = "conflicting_bias_induced"
            
            elif res['experiment_name'] == 'debias_female_sentences':
                if male_present and not female_present: outcome = "bias_flipped"
                elif not male_present and not female_present: outcome = "bias_neutralized"
                elif female_present and not male_present: outcome = "bias_persists"
                elif male_present and female_present: outcome = "conflicting_bias_induced"
            
            current_run_summary[key][outcome] = current_run_summary[key].get(outcome, 0) + 1

            sentence_report[key][outcome].append(original_sentence)

        existing_summary.update(current_run_summary)
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(existing_summary, f, indent=4, ensure_ascii=False)
        print(f"     - Saved summary of experiment outcomes to {summary_filename}")
        
        report_filename = os.path.join(steering_output_dir, "steering_sentence_report.json")
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(sentence_report, f, indent=4, ensure_ascii=False)
        print(f"     - Saved sentence-level report to {report_filename}")

        print("Cleaning unused entries from the API cache...")
        cleaned_cache = API_CACHE.get_cleaned_cache()
        SteeringCacheHelper.save_cache(CACHE_FILE, cleaned_cache)
        print(f"     - Saved {len(cleaned_cache)} used cache entries to {CACHE_FILE}")

    print("\nCorrected global concept steering experiment complete.")


if __name__ == "__main__":
    for _mode in Experiment.ANALYSIS_MODES:

        for experiment_id in range(10, 14):
            experiment_config = Experiment()
            experiment_config.EXPERIMENT_IDENTIFIER = experiment_id
            experiment_config.ANALYSIS_MODE = _mode

            API_CACHE = {}
            CACHE_FILE = None

            experiment_config.start()

            print(f"\n========================================================")
            print(f"==== RUNNING EXPERIMENT ID: {experiment_id}, MODE: {_mode} ====")
            print(f"========================================================\n")
            
            main(experiment_config)