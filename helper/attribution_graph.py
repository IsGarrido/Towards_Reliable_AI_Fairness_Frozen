import os
import json
import re
from typing import Dict, List, Any, Tuple, Optional, Set

class AttributionGraph:
    """
    A helper class to load, parse, and hold the data for a single attribution graph,
    with methods for traversing it.
    """

    def __init__(self, filepath: str):
        """Initializes the graph by loading, parsing, and processing the JSON file."""
        self.filepath: str = filepath
        self.filename: str = os.path.basename(filepath)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.links: List[Dict[str, Any]] = []
        self.logit_nodes: List[Dict[str, Any]] = []
        self.link_map: Dict[Tuple[str, str], float] = {}
        self.reverse_adjacency: Dict[str, List[str]] = {}
        
        self._load_and_process()

    def _parse_clerp(self, clerp_string: str) -> Tuple[Optional[str], Optional[float]]:
        """Parses the 'clerp' string to extract the predicted token and probability."""
        if not clerp_string:
            return None, None
        match = re.search(r'Output "([^"]+)" \(p=([=\d.<>]+)\)', clerp_string)
        if match:
            token = match.group(1)
            prob_str = match.group(2).replace('=', '').replace('<', '')
            try:
                probability = float(prob_str)
            except (ValueError, IndexError):
                probability = 0.0
            return token, probability
        return None, None

    def _load_and_process(self):
        """Loads raw data and builds necessary structures like reverse adjacency lists."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"   Error loading graph from {self.filepath}: {e}")
            return

        # Process nodes
        for node in raw_data.get("nodes", []):
            if node.get("feature_type") == "logit":
                token, prob = self._parse_clerp(node.get("clerp", ""))
                node["predicted_token"] = token
                node["token_probability"] = prob
                self.logit_nodes.append(node)
            self.nodes[node["node_id"]] = node
        
        # Process links to build reverse adjacency and a weight map
        self.links = raw_data.get("links", [])
        for link in self.links:
            source_id, target_id = link["source"], link["target"]
            weight = link.get("weight", 0.0)
            
            # For backtracking: target -> [source1, source2, ...]
            if target_id not in self.reverse_adjacency:
                self.reverse_adjacency[target_id] = []
            self.reverse_adjacency[target_id].append(source_id)
            
            # For easy weight lookup
            self.link_map[(source_id, target_id)] = weight
            
    def get_logit_node_for_token(self, token: str) -> Optional[str]:
        """Finds the node_id for a specific predicted token."""
        for node in self.logit_nodes:
            if node.get("predicted_token") == token:
                return node["node_id"]
        return None

    def backtrack_full(self, start_node_id: str) -> Tuple[Set[str], Set[Tuple[str, str]]]:
        """
        Performs a full backward traversal (DFS) from a given node, exploring all branches.
        Returns a set of all node_ids and a set of all (source, target) link tuples in the path.
        """
        path_nodes, path_links = set(), set()
        if start_node_id not in self.nodes:
            return path_nodes, path_links
        
        visited = set()
        stack = [start_node_id]

        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            
            visited.add(current_id)
            path_nodes.add(current_id)
            
            # Follow all predecessor links backwards
            if current_id in self.reverse_adjacency:
                for predecessor_id in self.reverse_adjacency[current_id]:
                    if predecessor_id not in visited:
                        stack.append(predecessor_id)
                        path_links.add((predecessor_id, current_id))
                        
        return path_nodes, path_links
