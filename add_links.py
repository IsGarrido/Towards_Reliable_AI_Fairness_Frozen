import os
import json
import glob
from urllib.parse import quote

def format_neuron_id_for_url(neuron_id: str) -> str:
    """
    Converts a neuron ID from the file's format (e.g., '24_397_8')
    to the format required by the Neuronpedia graph URL (e.g., '24_397_7').
    """
    parts = neuron_id.split('_')
    if len(parts) == 3:
        return f"{parts[0]}_{parts[1]}_7"
    return neuron_id

def update_neuron_links(file_path: str):
    """
    Reads a comparison JSON file, adds or updates cumulative "he/she" neuron 
    links, and overwrites the file, preserving all other existing links.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "differentiating_neurons" not in data or not isinstance(data.get("differentiating_neurons"), list):
            print(f"🟡 Info: Skipping '{file_path}' as it lacks a valid 'differentiating_neurons' list.")
            return

        base_url_he = "https://www.neuronpedia.org/gemma-2-2b/graph?slug=rel-exp_4-female_male-he_works_in_the_hospital_as_a&pruningThreshold=0.76&densityThreshold=0.99&pinnedIds="
        base_url_she = "https://www.neuronpedia.org/gemma-2-2b/graph?slug=rel-exp_4-female_male-she_works_in_the_hospital_as_a&pruningThreshold=0.76&densityThreshold=0.99&pinnedIds="
        url_prefix_to_replace = "https://www.neuronpedia.org/gemma-2-2b/graph?slug=rel-exp_4-female_male-"

        cumulative_url_ids = []
        for neuron in data.get("differentiating_neurons", []):
            original_neuron_id = neuron.get("neuron_id")
            if not original_neuron_id:
                continue

            url_neuron_id = format_neuron_id_for_url(original_neuron_id)
            cumulative_url_ids.append(url_neuron_id)
            pinned_ids_str = quote(",".join(cumulative_url_ids))

            existing_links = neuron.get("links", [])
            if not isinstance(existing_links, list):
                existing_links = []

            other_links = [
                link for link in existing_links 
                if not str(link).startswith(url_prefix_to_replace)
            ]

            new_he_link = base_url_he + pinned_ids_str
            new_she_link = base_url_she + pinned_ids_str
            
            neuron["links"] = other_links + [new_he_link, new_she_link]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Successfully processed and updated '{file_path}'")

    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON from '{file_path}'.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing '{file_path}': {e}")

def main():
    """
    Main function to find and process all 'comparison_*.json' files.
    """
    data_folder = "data"
    
    search_pattern = os.path.join(data_folder, '**', 'comparison_*.json')
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"\nNo files matching 'comparison_*.json' found in the '{data_folder}'.")
        return

    print(f"\nFound {len(json_files)} matching file(s). Starting update process...")
    for file_path in json_files:
        update_neuron_links(file_path)

    print("\nScript execution finished.")

if __name__ == "__main__":
    main()