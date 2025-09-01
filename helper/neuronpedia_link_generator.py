from typing import Dict, List


class NeuronpediaLinkGenerator:
    """
    Generates and adds Neuronpedia links to steering node objects.
    """
    _LINK_TEMPLATES = [
        "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{neuron}",
        # "https://www.neuronpedia.org/gemma-2-2b/{layer}-res-matryoshka-dc/{neuron}",
        # "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-att-16k/{neuron}",
        # "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-att-65k/{neuron}",
        # "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-mlp-65k/{neuron}",
        # "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{neuron}",
        # "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-65k/{neuron}",
    ]

    def _get_links(self, layer: int, neuron: int) -> List[str]:
        """
        Generates a list of Neuronpedia URLs for a specific neuron. (Private method)
        """
        return [
            template.format(layer=layer, neuron=neuron)
            for template in self._LINK_TEMPLATES
        ]

    def add_links(self, steering_node_data: Dict) -> Dict:
        """
        Parses the neuron_id from a steering node object and adds a 'links' array if it's a neuron.

        Args:
            steering_node_data: A dictionary representing a critical steering node.
                                It must contain a 'neuron_id' key.

        Returns:
            The steering_node_data dictionary, updated with a 'links' key if applicable.
        """
        neuron_id = steering_node_data.get("neuron_id")

        # Ensure neuron_id is a string and contains an underscore, which indicates a potential neuron ID
        if isinstance(neuron_id, str) and '_' in neuron_id:
            try:
                parts = neuron_id.split('_')
                # Ensure we have at least a layer and a neuron part (e.g., "25_11988" or "25_11988_4")
                if len(parts) >= 2:
                    layer = int(parts[0])
                    neuron = int(parts[1])
                    # Add the generated links to the dictionary
                    steering_node_data["links"] = self._get_links(layer=layer, neuron=neuron)
            except (ValueError, IndexError):
                pass

        return steering_node_data