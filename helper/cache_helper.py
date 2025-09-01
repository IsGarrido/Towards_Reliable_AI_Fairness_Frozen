import os
import json
import gzip 
from typing import Dict, Any, List

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from experiment import Experiment

class CacheHelper:
    """
    Manages loading, saving, and accessing API response caches.
    This version saves cache files in a compressed format (gzip).
    """

    def __init__(self, experiment: 'Experiment', all_sentence_groups: List[str]):
        """
        Initializes the CacheHelper.
        """
        self.experiment = experiment
        self.all_sentence_groups = all_sentence_groups
        self.global_cache_path = self.experiment.API_RESPONSE_CACHE_FILEPATH
        self.experiment_cache_path = self._get_experiment_cache_path()
        
        self.global_cache: Dict[str, Any] = {}
        self.experiment_cache: Dict[str, Any] = {}
        self.cache_changed = False

        self._load_caches()

    def _get_experiment_cache_path(self) -> str:
        """Constructs the file path for the compressed experiment-specific cache."""
        base_dir = self.experiment.sentence_provider.get_data_dir()
        filename = f"cache_{self.experiment.EXPERIMENT_IDENTIFIER}.json.gz"
        return os.path.join(base_dir, filename)

    def _load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Safely loads a JSON file, handling both compressed and uncompressed files."""
        if not os.path.exists(file_path):
            return {}
        
        try:
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError, gzip.BadGzipFile) as e:
            print(f"Warning: Could not load or parse cache file '{file_path}': {e}")
            return {}

    def _save_json_file(self, data: Dict[str, Any], file_path: str):
        """Saves a dictionary to a JSON file, using gzip compression if specified."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error: Could not save cache file '{file_path}': {e}")

    # The rest of the methods ( _load_caches, get_key, get, set, save_caches )
    # do not need to be changed, as they call the modified I/O methods.
    
    def _load_caches(self):
        """
        Loads the global and experiment caches. If the experiment cache doesn't
        exist, it's created and seeded from the global cache.
        """
        print("Loading caches...")
        self.global_cache = self._load_json_file(self.global_cache_path)
        
        if os.path.exists(self.experiment_cache_path):
            print(f"Found existing experiment cache: {self.experiment_cache_path}")
            self.experiment_cache = self._load_json_file(self.experiment_cache_path)
        else:
            print(f"No cache found for experiment '{self.experiment.EXPERIMENT_IDENTIFIER}'. Seeding from global cache...")
            self.experiment_cache = {}
            sentence_provider = self.experiment.sentence_provider
            
            for sentences in sentence_provider.get_data().values():
                for sentence in sentences:
                    slug = sentence_provider.generate_slug(
                        self.all_sentence_groups,
                        sentence,
                        self.experiment.EXPERIMENT_IDENTIFIER
                    )
                    key = self.get_key(sentence, slug)
                    if key in self.global_cache:
                        self.experiment_cache[key] = self.global_cache[key]
            
            if self.experiment_cache:
                print(f"Seeded {len(self.experiment_cache)} items from global cache.")
                self.cache_changed = True

    def get_key(self, sentence: str, slug: str) -> str:
        """Generates a unique cache key."""
        return f"{self.experiment.MODEL_IDENTIFIER}:{slug}:{sentence}"

    def get(self, key: str) -> Any:
        """Retrieves an item from the experiment cache."""
        return self.experiment_cache.get(key)

    def set(self, key: str, value: Any):
        """Adds or updates an item in the experiment cache and marks it as changed."""
        self.experiment_cache[key] = value
        self.cache_changed = True

    def save_caches(self):
        """
        Saves the experiment cache (compressed) and updates the global cache if changes were made.
        """
        if not self.cache_changed:
            print("\nNo new data fetched. Caches are up to date.")
            return

        print("\nSaving changes to caches...")
        
        print(f"Saving compressed experiment cache to: {self.experiment_cache_path}")
        self._save_json_file(self.experiment_cache, self.experiment_cache_path)
        
        print(f"Updating and saving global cache to: {self.global_cache_path}")
        self.global_cache.update(self.experiment_cache)
        self._save_json_file(self.global_cache, self.global_cache_path)
        
        self.cache_changed = False