import json
from typing import Any, Dict, List, Tuple
from helper.sentence_provider import SentenceDataProvider


class SteeringFilterHelper:
    """
    A helper class to filter sentences based on specific criteria before steering.
    It preloads and caches common data, like sentence attribution graphs, for efficiency.
    """
    def __init__(self, sentence_provider: SentenceDataProvider, data_dir: str, all_groups: List[str], experiment_id: int):
        """
        Initializes the helper with necessary data providers and configuration.
        """
        self.sentence_provider = sentence_provider
        self.data_dir = data_dir
        self.all_groups = all_groups
        self.experiment_id = experiment_id
        self.graph_cache: Dict[str, Dict] = {} # Cache for sentence graphs

    def _load_json_file(self, filepath: str) -> Dict:
        """Loads a JSON file and returns its content."""
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load or parse {filepath}.")
            return None

    def _get_sentence_graph(self, sentence: str) -> Dict:
        """
        Retrieves a sentence's attribution graph, using a cache to avoid redundant loads.
        """
        if sentence in self.graph_cache:
            return self.graph_cache[sentence]

        slug = self.sentence_provider.generate_slug(self.all_groups, sentence, self.experiment_id)
        graph_filepath = os.path.join(self.data_dir, f"{slug}.json")
        graph = self._load_json_file(graph_filepath)

        if graph:
            self.graph_cache[sentence] = graph
        return graph

    def get_output_tokens_from_graph(self, graph: Dict[str, Any]) -> List[Tuple[str, float, str]]:
        """
        Parses a graph to find all output tokens and their probabilities.
        """
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

    def filter_by_token_possibility(self, sentences_to_filter: List[Dict], source_token: str, target_token: str, show_prints: bool = True) -> List[Dict]:
        """
        Filters a list of sentences, keeping only those where both source and target tokens are possible outputs.
        """
        filtered_list = []
        source_token_clean = source_token.strip().lower()
        target_token_clean = target_token.strip().lower()
        
        for sentence_info in sentences_to_filter:
            sentence = sentence_info.get("sentence")
            if not sentence:
                continue

            graph = self._get_sentence_graph(sentence)
            possible_outputs_raw = self.get_output_tokens_from_graph(graph)
            
            if not possible_outputs_raw:
                if show_prints:
                    print(f"\033[93m           - Skipping steer: No output tokens found in graph for sentence \"{sentence[:70]}...\".\033[0m")
                continue

            possible_outputs_set = {token_info[0].strip().lower() for token_info in possible_outputs_raw}
            source_is_possible = source_token_clean in possible_outputs_set
            target_is_possible = target_token_clean in possible_outputs_set

            if source_is_possible and target_is_possible:
                filtered_list.append(sentence_info)
            elif show_prints:
                print(f"\033[93m           - Skipping steer for sentence: \"{sentence[:70]}...\"\033[0m")
                if not source_is_possible:
                    print(f"\033[93m             - Reason: Source token '{source_token}' not in the graph's possible outputs.\033[0m")
                if not target_is_possible:
                    print(f"\033[93m             - Reason: Target token '{target_token}' not in the graph's possible outputs.\033[0m")
        
        return filtered_list

    def filter_by_most_likely_token(self, sentences_to_filter: List[Dict], source_token: str, show_prints: bool = True) -> List[Dict]:
        """
        Filters a list of sentences, keeping only those where the source token is the most probable output.
        """
        filtered_list = []
        source_token_clean = source_token.strip().lower()

        for sentence_info in sentences_to_filter:
            sentence = sentence_info.get("sentence")
            if not sentence:
                continue
            
            graph = self._get_sentence_graph(sentence)
            possible_outputs_raw = self.get_output_tokens_from_graph(graph)

            if not possible_outputs_raw:
                continue

            most_likely_output = max(possible_outputs_raw, key=lambda item: item[1])
            most_likely_token_text = most_likely_output[0].strip().lower()

            if most_likely_token_text == source_token_clean:
                filtered_list.append(sentence_info)
            elif show_prints:
                source_prob = next((p for t, p, _ in possible_outputs_raw if t.strip().lower() == source_token_clean), 0.0)
                print(f"\033[93m           - Skipping steer for sentence: \"{sentence[:70]}...\"\033[0m")
                print(f"\033[93m             - Reason: Source token '{source_token}' (p={source_prob:.4f}) is not the most likely. Most likely: '{most_likely_output[0]}' (p={most_likely_output[1]:.4f}).\033[0m")
                
        return filtered_list