import requests
import time
import json
from typing import Dict

class SteeringHelper:
    """
    A helper class to manage interaction with the steering API, including
    making calls, handling errors, and formatting data.
    """
    def __init__(self, api_key: str, api_url: str, model_id: str, cache_ref: Dict):
        self.api_key = api_key
        self.api_url = api_url
        self.model_id = model_id
        self.cache = cache_ref
        self.min_seconds_per_request = 3600 / 300  # 300 requests per hour = 12 seconds/request
        self.last_api_call_time = None

    def _handle_api_error(self, e: requests.exceptions.RequestException, payload: Dict) -> Dict:
        """
        Handles different types of requests.exceptions.RequestException and
        extracts relevant error information.
        """
        if isinstance(e, requests.exceptions.HTTPError):
            status_code = e.response.status_code
            reason_phrase = e.response.reason
            error_message_prefix = f"   - API call failed with HTTP Error: {status_code} {reason_phrase}"
            bad_request_reason = None
            try:
                error_details = e.response.json()
                if isinstance(error_details, dict):
                    if 'message' in error_details: bad_request_reason = error_details['message']
                    elif 'error' in error_details:
                        if isinstance(error_details['error'], str): bad_request_reason = error_details['error']
                        elif isinstance(error_details['error'], dict) and 'message' in error_details['error']: bad_request_reason = error_details['error']['message']
                    elif 'detail' in error_details: bad_request_reason = error_details['detail']
                    elif 'reason' in error_details: bad_request_reason = error_details['reason']
                    if bad_request_reason is None: bad_request_reason = json.dumps(error_details, indent=2)
            except json.JSONDecodeError:
                bad_request_reason = e.response.text.strip()
                if not bad_request_reason: bad_request_reason = "No detailed error message provided in response body."
            
            if bad_request_reason: print("\033[91m" + f"{error_message_prefix}\n   - Bad Request Reason: {bad_request_reason}" + "\033[0m")
            else: print("\033[91m" + error_message_prefix + "\033[0m")
            return {"error": f"{error_message_prefix} | Reason: {bad_request_reason}"}
        else:
            print("\033[91m" + f"   - API call failed due to network/timeout error: {e}" + "\033[0m")
            return {"error": str(e)}

    def call_api(self, payload: Dict, save_cache_func) -> Dict:
        """Makes a POST request to the steering API with caching and rate limiting."""
        cache_key = json.dumps(payload, sort_keys=True)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if isinstance(cached_result, dict) and "error" not in cached_result:
                print(f"\033[92mUsing cached result for payload hash: {hash(cache_key)}\033[0m")
                return cached_result
            else:
                del self.cache[cache_key]
                print(f"\033[93mFound a previously failed request in cache. Retrying payload hash: {hash(cache_key)}\033[0m")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        try:
            if self.last_api_call_time:
                elapsed = time.time() - self.last_api_call_time
                if elapsed < self.min_seconds_per_request:
                    wait_duration = self.min_seconds_per_request - elapsed
                    print(f"\nRate limit: Waiting for {wait_duration:.2f} seconds...")
                    time.sleep(wait_duration)
            
            self.last_api_call_time = time.time() 
            json_payload = json.dumps(payload, ensure_ascii=False, indent=2)
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            self.cache[cache_key] = result
            save_cache_func() 
            return result
        
        except requests.exceptions.RequestException as e:
            print("\033[94m" + f"   - Called API with payload: {json_payload}" + "\033[0m")
            error_result = self._handle_api_error(e, payload)
            return error_result
        except Exception as e:
            print("\033[91m" + f"   - An unexpected error occurred: {e}" + "\033[0m")
            return {"error": str(e)}

    def as_feature_nid(self, neuron_id: str, strength: float) -> dict:
        """Formats a neuron ID into the API's expected feature dictionary."""
        parts = neuron_id.split('_')
        if len(parts) >= 2:
            layer, index = parts[0], parts[1]
            return {"modelId": self.model_id, "layer": f"{layer}-gemmascope-res-16k", "index": int(index), "strength": strength}
        return None