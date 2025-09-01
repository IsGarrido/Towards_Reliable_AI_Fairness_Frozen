import csv
import os
import requests
import time
import json
from typing import Dict, List, Any
from helper.sentence_provider import SentenceDataProvider
from experiment import Experiment
from helper.cache_helper import CacheHelper # Import the new helper

# --- Configuration ---
current_experiment = Experiment()

# --- Setup from Experiment Configuration ---
DATA_DIR = current_experiment.sentence_provider.get_data_dir()
SENTENCE_GROUPS: Dict[str, List[str]] = current_experiment.sentence_provider.get_data()

# Write sentence files for reference
for group_name, sentences in SENTENCE_GROUPS.items():
    print(f"Group '{group_name}' has {len(sentences)} sentences.")
    output_filename = os.path.join(DATA_DIR, f"sentences_{group_name}.tsv")
    print(f"Writing {len(sentences)} sentences for group '{group_name}' to '{output_filename}'...")
    try:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(['sentence'])
            for sentence in sentences:
                writer.writerow([sentence])
    except IOError as e:
        print(f"Error: Could not write to file {output_filename}. Reason: {e}")

# --- Helper Functions ---
def ensure_dir_exists(path: str):
    """Ensures that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    print(f"Directory '{path}' is ready.")

def generate_and_download_graph(sentence: str, slug: str, cache_handler: CacheHelper) -> bool:
    """
    Generates and downloads an attribution graph, using the cache handler.
    """
    print(f"\n--- Generating graph for slug: {slug} ---")

    cache_key = cache_handler.get_key(sentence, slug)
    cached_item = cache_handler.get(cache_key)

    if cached_item:
        print(f"Found cached response for '{sentence}'. Using cached data.")
        try:
            file_path = os.path.join(DATA_DIR, f"{slug}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cached_item['graph_data'], f, indent=2)
            print(f"Successfully saved graph data from cache to: {file_path}")
            return True
        except Exception as e:
            print(f"Error using cached data: {e}. Will make a new API request.")

    if not current_experiment.API_KEY or "YOUR_API_KEY" in current_experiment.API_KEY:
        print("Error: API_KEY is not set in the Experiment class.")
        return False

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {current_experiment.API_KEY}"}
    payload = Experiment.get_graph_generation_payload(sentence, slug)

    try:
        print(f"Sending request to Neuronpedia API for prompt: '{sentence}'")
        response = requests.post(
            f"{current_experiment.API_BASE_URL}/generate",
            headers=headers,
            json=payload,
            timeout=current_experiment.API_POST_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        response_data = response.json()
        print("API Response Message:", response_data.get("message"))
        
        s3_url = response_data.get("s3url")
        if not s3_url:
            print(f"Error: No S3 URL received for slug '{slug}'.")
            return False
            
        s3_response = requests.get(s3_url, timeout=current_experiment.API_DOWNLOAD_REQUEST_TIMEOUT_SECONDS)
        s3_response.raise_for_status()
        graph_data = s3_response.json()
        
        new_cache_entry = {
            "timestamp": time.time(), "api_response": response_data,
            "graph_data": graph_data, "sentence": sentence, "slug": slug
        }
        cache_handler.set(cache_key, new_cache_entry)
        
        file_path = os.path.join(DATA_DIR, f"{slug}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
            
        print(f"Successfully saved graph data to: {file_path}")
        return True

    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 409:
            print(f"Graph with slug '{slug}' already exists on server. Retrieving it.")
            try:
                get_url = f"{current_experiment.API_BASE_URL}/{current_experiment.MODEL_IDENTIFIER}/{slug}"
                get_response = requests.get(
                    get_url,
                    headers=headers,
                    timeout=current_experiment.API_DOWNLOAD_REQUEST_TIMEOUT_SECONDS
                )
                get_response.raise_for_status()
                retrieved_data = get_response.json()
                s3_url = retrieved_data.get("url")

                if not s3_url:
                    print(f"Error: Retrieved existing graph for '{slug}', but no S3 URL was found.")
                    return False

                s3_response = requests.get(s3_url, timeout=current_experiment.API_DOWNLOAD_REQUEST_TIMEOUT_SECONDS)
                s3_response.raise_for_status()
                graph_data = s3_response.json()

                # Cache the successfully retrieved data
                existing_cache_entry = {
                    "timestamp": time.time(), "api_response": retrieved_data,
                    "graph_data": graph_data, "sentence": sentence, "slug": slug
                }
                cache_handler.set(cache_key, existing_cache_entry)

                file_path = os.path.join(DATA_DIR, f"{slug}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2)
                
                print(f"Successfully saved existing graph data to: {file_path}")
                return True
            except requests.exceptions.RequestException as get_e:
                print(f"Error: Failed to retrieve the existing graph for '{slug}'. Reason: {get_e}")
                return False
        
        elif e.response is not None:
            print(f"An API error occurred for slug '{slug}': {e}\nError body: {e.response.text}")
        else:
            print(f"A network error occurred for slug '{slug}': {e}")
        
        time.sleep(current_experiment.DELAY_AFTER_FAILED_REQUEST_SECONDS)
        return False

# --- Main Execution ---
def main():
    """Main function to orchestrate the graph generation and download process."""
    print("Starting Step 1: Generate and Download Attribution Graphs")
    ensure_dir_exists(DATA_DIR)
    
    all_groups = list(SENTENCE_GROUPS.keys())
    cache_handler = CacheHelper(current_experiment, all_groups)

    for group_name, sentences in SENTENCE_GROUPS.items():
        for sentence in sentences:
            slug = SentenceDataProvider.generate_slug(
                all_groups, sentence, current_experiment.EXPERIMENT_IDENTIFIER
            )

            if os.path.exists(os.path.join(DATA_DIR, f"{slug}.json")):
                print(f"\nSkipping generation for '{slug}' as file already exists.")
                continue

            success = generate_and_download_graph(sentence, slug, cache_handler)
            if success:
                delay = current_experiment.DELAY_BETWEEN_SUCCESSFUL_REQUESTS_SECONDS
                print(f"Waiting {delay} seconds before the next request...")
                time.sleep(delay)

    cache_handler.save_caches()
    print("\nGraph generation and download process finished.")

if __name__ == "__main__":
    main()