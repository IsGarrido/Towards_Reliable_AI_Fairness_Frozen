import json
import time
from typing import Dict

import requests


class SteerHelper:
    """
    Manages API interactions, caching, and data preparation for steering.
    """
    def __init__(self, api_key: str, model_id: str, api_url: str, cache_file: str):
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = api_url
        self.cache_file = cache_file
        self.api_cache = {}
        self.load_api_cache()

    def load_api_cache(self):
        """Loads the API cache from the specified JSON file if it exists."""
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.api_cache = json.load(f)
                print(f"Loaded {len(self.api_cache)} items from cache file: {self.cache_file}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load cache file. Starting with a new cache. Error: {e}")
        else:
            print("No cache file found. Starting with a new cache.")

    def save_api_cache(self):
        """Saves the API cache to the specified JSON file."""
        if self.cache_file:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.api_cache, f, indent=4)
                print(f"Saved API cache with {len(self.api_cache)} items.")
            except IOError as e:
                print(f"Error: Could not save cache file. Error: {e}")

    def call_api(self, payload: Dict) -> Dict:
        """Makes a request to the Neuronpedia steering API and caches successful responses."""
        cache_key = json.dumps(payload, sort_keys=True)
        if cache_key in self.api_cache:
            return self.api_cache[cache_key]

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        try:
            time.sleep(2)  # Wait before making the API call
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            self.api_cache[cache_key] = result
            return result
        except requests.exceptions.RequestException as e:
            print("\033[91m" + f"     - API call failed: {e}" + "\033[0m")
            print("# --- BEGIN FAILED PAYLOAD ---")
            print(json.dumps(payload, indent=4))
            print("# --- END FAILED PAYLOAD ---")
            return {"error": str(e), "status_code": response.status_code if 'response' in locals() else None}

    @staticmethod
    def parse_neuron_id(neuron_id: str) -> dict:
        """Parses 'layer_index_...' string into components for the API."""
        parts = neuron_id.split('_')
        if len(parts) >= 2:
            try:
                layer, index = parts[0], parts[1]
                layer_str = f"{layer}-gemmascope-res-16k"
                return {"layer": layer_str, "index": int(index)}
            except (ValueError, IndexError):
                return None
        return None

    def as_feature_nid(self, neuron_id: str, strength: float) -> dict:
        """Converts a neuron ID and strength into a feature payload for the API."""
        parsed = self.parse_neuron_id(neuron_id)
        if parsed is None:
            return None
        return {"modelId": self.model_id, "layer": parsed["layer"], "index": parsed["index"], "strength": strength}

    @staticmethod
    def evaluate_steering(result: Dict, source_token: str, target_token: str) -> Dict:
        """Checks if the steering was successful based on the STEERED property."""
        steered_text = result.get("STEERED", "").lower()
        source_token_clean = source_token.strip().lower()
        target_token_clean = target_token.strip().lower()
        source_is_gone = source_token_clean not in steered_text
        target_appeared = target_token_clean in steered_text
        return {
            "success": source_is_gone and target_appeared,
            "source_token_gone": source_is_gone,
            "target_token_appeared": target_appeared,
            "generated_text": steered_text,
            "shareUrl": result.get("shareUrl")
        }