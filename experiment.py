from helper.sentence_provider import SentenceDataProvider
from typing import Dict, Any, List

class Experiment:
    # --- Primary Experiment Settings ---
    # A unique integer identifier for this specific experiment run.
    EXPERIMENT_IDENTIFIER: int = 13
    # A string describing the analysis mode, e.g., "weighted_difference".
    ANALYSIS_MODE: str = "weighted_difference"

    # --- API and Model Configuration ---
    # The base URL for the Neuronpedia API.
    API_BASE_URL: str = "https://www.neuronpedia.org/api/graph"
    # The identifier for the model to be used in the experiment (e.g., "gemma-2-2b").
    MODEL_IDENTIFIER: str = "gemma-2-2b"
    # Your private API key for accessing the Neuronpedia API.
    API_KEY: str = "sk-np-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

    # --- File and Directory Paths ---
    # The file path for the global, shared API response cache.
    API_RESPONSE_CACHE_FILEPATH: str = "data/api_response_cache.json"

    # --- Graph Generation API Parameters ---
    # Parameters sent in the body of the POST request to generate a graph.
    GRAPH_GENERATION_PAYLOAD_PARAMETERS: Dict[str, Any] = {
        "maxNLogits": 15,
        "desiredLogitProb": 0.9,
        "nodeThreshold": 0.7,
        "edgeThreshold": 0.8,
        "maxFeatureNodes": 10000
    }
    # Timeout in seconds for the initial graph generation POST request.
    API_POST_REQUEST_TIMEOUT_SECONDS: int = 120
    # Timeout in seconds for downloading data (e.g., from S3 or GET requests).
    API_DOWNLOAD_REQUEST_TIMEOUT_SECONDS: int = 60
    # Delay in seconds after a successful API request to respect rate limits.
    DELAY_BETWEEN_SUCCESSFUL_REQUESTS_SECONDS: int = 10
    # Delay in seconds after a failed API request before moving on.
    DELAY_AFTER_FAILED_REQUEST_SECONDS: int = 10
    
    # --- Analysis-Specific Parameters ---
    # The minimum number of times a token must appear in both groups to be analyzed.
    ANALYSIS_MIN_TOKEN_FREQUENCY_PER_GROUP: int = 5
    # The minimum number of times a neuron must connect in both groups to be analyzed.
    ANALYSIS_MIN_NEURON_CONNECTIONS_PER_GROUP: int = 5
    # The number of top differentiating neurons to include in the final output.
    ANALYSIS_TOP_N_NEURONS_TO_SAVE: int = 100
    # A small constant for numerical stability in calculations.
    ANALYSIS_CALCULATION_EPSILON: float = 1e-6
    # The prefix used to identify and exclude embedding layer neurons.
    ANALYSIS_EMBEDDING_NEURON_PREFIX: str = "0_"

    # --- Neuronpedia Link Configuration ---
    # URL templates for generating links to Neuronpedia.
    NEURONPEDIA_LINK_TEMPLATES: List[str] = [
        "https://www.neuronpedia.org/{model}/{layer}-gemmascope-transcoder-16k/{neuron}",
    ]

    # --- Output Filename Templates ---
    # The current, simplified template for the analysis sub-directory.
    OUTPUT_ANALYSIS_SUBDIR_TEMPLATE: str = "{analysis_name}"
    # Filename for the token probability analysis output.
    OUTPUT_TOKEN_ANALYSIS_FILENAME: str = "token_probability_analysis.json"
    # Filename for the main global comparison output.
    OUTPUT_GLOBAL_COMPARISON_FILENAME: str = "comparison_global_male_biased_vs_global_female_biased.json"
    
    # --- Deprecated Templates (For Backward Compatibility) ---
    # Old template for the differential analysis directory name.
    OUTPUT_ANALYSIS_DIR_TEMPLATE_OLD: str = "__differential_analysis_{analysis_name}"
    # Old template for the specific experiment output sub-directory.
    OUTPUT_EXPERIMENT_SUBDIR_TEMPLATE_OLD: str = "exp{experiment_id}_cross_comparison_{analysis_name}"

    ANALYSIS_MODES = [
        "absolute_difference",
        # "bhattacharyya",
        # "bhattacharyya_normalized",
        "bhattacharyya_specificity",
        "bidirectional_specificity",
        # "mean_weight_difference",
        "weighted_difference",
    ]

    # --- Steering Experiment Parameters ---
    # The URL for the Neuronpedia steering API endpoint.
    STEERING_API_URL: str = "https://www.neuronpedia.org/api/steer"
    # The specific analysis mode this steering experiment should read results from.
    STEERING_INPUT_ANALYSIS_MODE: str = "weighted_difference"
    # Parameter sweep: A list of the number of top neurons to use for steering.
    STEERING_N_NEURONS_SWEEP: List[int] = [1, 3, 5, 10]
    # Parameter sweep: A list of multipliers to apply to steering vectors.
    STEERING_MULTIPLIER_SWEEP: List[float] = [1, 10, 50]
    # A global multiplier applied to all steering strengths.
    STEERING_GLOBAL_STRENGTH_MULTIPLIER: float = 1.0
    # The layer name format required by the steering API's `as_feature_nid` function.
    STEERING_LAYER_FORMAT: str = "{layer}-gemmascope-res-16k"
    # A fixed seed for reproducibility in steering API calls.
    STEERING_REQUEST_SEED: int = 42
    # Default parameters for the steering API payload.
    STEERING_API_DEFAULT_PARAMS: Dict[str, Any] = {
        "temperature": 0.7,
        "n_tokens": 10,
        "freq_penalty": 1.0,
    }
    # Timeout in seconds for steering API requests.
    STEERING_API_TIMEOUT_SECONDS: int = 90
    # Delay in seconds between steering API requests.
    STEERING_API_DELAY_SECONDS: float = 2.0
    # The sub-directory where steering results will be saved.
    STEERING_OUTPUT_SUBDIR_NAME: str = "steering_experiments_global_concept_corrected"
    # The filename for the local cache of steering API calls.
    STEERING_CACHE_FILENAME: str = "api_request_cache.json"
    # The filename for the raw, detailed results of the steering runs.
    STEERING_RAW_RESULTS_FILENAME: str = "steering_raw_results.json"
    # The filename for the summarized results of the steering runs.
    STEERING_SUMMARY_FILENAME: str = "steering_summary.json"


    def __init__(self):
        self.start()

    def start(self):
        """
        Initializes the experiment configuration and sets up the
        correct sentence provider based on the EXPERIMENT_IDENTIFIER.
        """
        if self.EXPERIMENT_IDENTIFIER == 10:
            self.sentence_provider = SentenceDataProvider(
                self.EXPERIMENT_IDENTIFIER,
                use_sample=False,
                use_adjetives=False,
                use_nursedoctor=False,
                use_boring_nd=False,
                use_prof=True)
        elif self.EXPERIMENT_IDENTIFIER == 11:
            self.sentence_provider = SentenceDataProvider(
                self.EXPERIMENT_IDENTIFIER,
                use_sample=False,
                use_adjetives=True,
                use_nursedoctor=False,
                use_boring_nd=False,
                use_prof=False)
        elif self.EXPERIMENT_IDENTIFIER == 12:
            self.sentence_provider = SentenceDataProvider(
                self.EXPERIMENT_IDENTIFIER,
                use_sample=False,
                use_adjetives=False,
                use_nursedoctor=False,
                use_boring_nd=True,
                use_prof=False)
        elif self.EXPERIMENT_IDENTIFIER == 13:
            self.sentence_provider = SentenceDataProvider(
                self.EXPERIMENT_IDENTIFIER,
                use_sample=False,
                use_adjetives=False,
                use_nursedoctor=True,
                use_boring_nd=False,
                use_prof=False)
        else:
            raise ValueError(f"No configuration found for EXPERIMENT_IDENTIFIER: {self.EXPERIMENT_IDENTIFIER}")
        

    @staticmethod
    def get_graph_generation_payload(prompt_sentence: str, graph_slug: str) -> Dict[str, Any]:
        """
        Constructs the full payload for the graph generation API request
        using the class-level configuration parameters.
        """
        payload = {
            "prompt": prompt_sentence,
            "modelId": Experiment.MODEL_IDENTIFIER,
            "slug": graph_slug,
        }
        payload.update(Experiment.GRAPH_GENERATION_PAYLOAD_PARAMETERS)
        return payload