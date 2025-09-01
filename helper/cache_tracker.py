from typing import Dict, Set


class CacheTracker(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used_keys: Set[str] = set()

    def __getitem__(self, key):
        self.used_keys.add(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.used_keys.add(key)
        super().__setitem__(key, value)
    
    def get(self, key, default=None):
        if key in self:
            self.used_keys.add(key)
        return super().get(key, default)

    def get_cleaned_cache(self) -> Dict:
        return {key: super().get(key) for key in self.used_keys}