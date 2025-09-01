import os
import json
import gzip
from typing import Dict, Any

class SteeringCacheHelper:
    """
    A helper to load and save the API cache for steering experiments.
    It saves caches in a compressed format (json.gz) and can read both the
    new compressed format and the old uncompressed JSON format for backward
    compatibility.
    """

    @staticmethod
    def load_cache(file_path: str) -> Dict[str, Any]:
        """
        Loads the API cache. It first tries to load from the new compressed
        format (.json.gz). If that file doesn't exist, it falls back to
        the old uncompressed format (.json).
        """
        if not file_path:
            return {}

        new_format_path = file_path + '.gz'
        old_format_path = file_path

        # Prioritize loading the new, compressed format
        if os.path.exists(new_format_path):
            try:
                with gzip.open(new_format_path, 'rt', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                print(f"Loaded {len(loaded_cache)} items from compressed cache: {new_format_path}")
                return loaded_cache
            except (json.JSONDecodeError, IOError, gzip.BadGzipFile) as e:
                print(f"Warning: Could not load compressed cache file '{new_format_path}'. Error: {e}")

        # Fallback to loading the old, uncompressed format
        if os.path.exists(old_format_path):
            try:
                with open(old_format_path, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                print(f"Loaded {len(loaded_cache)} items from old format cache: {old_format_path}")
                return loaded_cache
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load old format cache file '{old_format_path}'. Error: {e}")
        
        # Return an empty dict if no cache file is found or if loading fails
        return {}

    @staticmethod
    def save_cache(file_path: str, data: Dict[str, Any]):
        """Saves the API cache to the new compressed JSON format (.json.gz)."""
        if not file_path:
            return

        new_format_path = file_path + '.gz'
        
        try:
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(new_format_path), exist_ok=True)
            
            # Save data in the new compressed format
            with gzip.open(new_format_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} items to compressed cache: {new_format_path}")

        except IOError as e:
            print(f"Error: Could not save cache file to '{new_format_path}'. Error: {e}")